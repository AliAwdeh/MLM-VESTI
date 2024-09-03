import pandas as pd
import numpy as np
import os
import faiss
from sklearn.preprocessing import OneHotEncoder
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
import pickle

# Load CSV data
styles_df = pd.read_csv('styles.csv')

# Drop unnecessary columns
styles_df = styles_df.drop(['id', 'productDisplayName'], axis=1)

# Encode categorical data
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(styles_df[['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage']]).toarray()

# Initialize the ResNet50 model pre-trained on ImageNet
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

# Directory containing images
image_dir = 'images'

# Directory for storing model versions locally
version_dir = 'model_versions'

# Ensure the version directory exists
if not os.path.exists(version_dir):
    os.makedirs(version_dir)

# Function to extract image features
def extract_image_features(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()

# Function to get the latest version
def get_latest_version():
    existing_versions = [int(f.replace('mlmv', '')) for f in os.listdir(version_dir) if f.startswith('mlmv')]
    if not existing_versions:
        return None
    latest_version = max(existing_versions)
    return f"mlmv{latest_version}"

# Function to get the next version
def get_next_version():
    existing_versions = [int(f.replace('mlmv', '')) for f in os.listdir(version_dir) if f.startswith('mlmv')]
    next_version = max(existing_versions, default=-1) + 1
    return f"mlmv{next_version}"

# Function to list available versions
def list_versions():
    versions = [f for f in os.listdir(version_dir) if f.startswith('mlmv')]
    return sorted(versions, key=lambda x: int(x.replace('mlmv', '')))

# Function to load a specific model version
def load_model(version):
    global index
    global combined_features
    global encoder

    version_path = os.path.join(version_dir, version)
    
    # Load the index and features from the local storage
    index = faiss.read_index(os.path.join(version_path, f'{version}_index.bin'))
    combined_features = np.load(os.path.join(version_path, f'{version}_features.npy'))

    with open(os.path.join(version_path, f'{version}_encoder.pkl'), 'rb') as f:
        encoder = pickle.load(f)

    print(f"Model version {version} loaded successfully from local storage!")

# Function to train the model and save the new version locally
def train_model():
    # Check for the latest version
    latest_version = get_latest_version()
    
    if latest_version:
        print(f"Loading the latest model version: {latest_version}")
        load_model(latest_version)
    else:
        print("No previous model found. Initializing a new model.")
        dimension = encoded_features.shape[1] + 2048  # Metadata + image feature dimensions
        global index  # Initialize global index
        index = faiss.IndexFlatL2(dimension)
        global combined_features  # Initialize global combined features
        combined_features = np.empty((0, dimension))

    image_features = []
    num_images = len(styles_df)
    
    for i, image_name in enumerate(styles_df.index):
        image_path = os.path.join(image_dir, str(image_name) + '.jpg')
        if os.path.exists(image_path):
            features = extract_image_features(image_path)
            image_features.append(features)
        else:
            image_features.append(np.zeros(2048))  # Assuming 2048-dimensional feature vectors from ResNet
        
        # Report progress
        if (i + 1) % 100 == 0 or (i + 1) == num_images:
            print(f"Processed {(i + 1) / num_images * 100:.2f}% of images")

    # Combine image features with encoded metadata features
    new_combined_features = np.hstack((encoded_features, np.array(image_features)))

    # Add new features to the existing index
    index.add(new_combined_features.astype('float32'))

    # Update combined features
    combined_features = np.vstack((combined_features, new_combined_features))

    # Save the new model version locally
    version = get_next_version()
    version_path = os.path.join(version_dir, version)
    os.makedirs(version_path)
    
    faiss.write_index(index, os.path.join(version_path, f'{version}_index.bin'))
    np.save(os.path.join(version_path, f'{version}_features.npy'), combined_features)
    
    with open(os.path.join(version_path, f'{version}_encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)

    print(f"Model trained and saved successfully as version {version} on this device!")

# Function to find similar items
def find_similar_items(query_vector, k=5):
    distances, indices = index.search(query_vector.reshape(1, -1), k)
    return indices, distances

# Function to test the model
def test_model(image_path, version):
    load_model(version)
    
    query_features = extract_image_features(image_path)
    query_vector = np.hstack((np.zeros(encoded_features.shape[1]), query_features))
    
    similar_indices, _ = find_similar_items(query_vector, k=5)

    # Count the number of similar items found
    num_similar_items = len(similar_indices[0])

    print("Similar items found:")
    for idx in similar_indices[0]:
        print(styles_df.iloc[idx])

    # Print the number of similar items found
    print(f"\nNumber of similar items found: {num_similar_items}")

    # Calculate accuracy based on articleType match count
    query_article_type = styles_df.loc[styles_df.index == int(os.path.basename(image_path).split('.')[0]), 'articleType'].values[0]
    predicted_article_types = styles_df.iloc[similar_indices[0]]['articleType'].values
    matches = sum(predicted_article_types == query_article_type)
    accuracy = (matches / num_similar_items) * 100
    print(f"Accuracy: {accuracy:.2f}% based on {matches} out of {num_similar_items} similar items matching the articleType")

# Function to handle user input and control the flow
def main():
    while True:
        print("\nChoose an option:")
        print("1. Train the model")
        print("2. Test the model on a picture")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            train_model()
        elif choice == '2':
            versions = list_versions()
            if not versions:
                print("No versions available. Please train the model first.")
            else:
                print("Available versions:")
                for version in versions:
                    print(version)
                selected_version = input("Enter the version to use (e.g., mlmv0): ")
                if selected_version in versions:
                    image_path = input("Enter the path of the image to test: ")
                    if os.path.exists(image_path):
                        test_model(image_path, selected_version)
                    else:
                        print("Image not found, please check the path.")
                else:
                    print("Invalid version selected.")
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")

# Run the program
if __name__ == "__main__":
    main()
