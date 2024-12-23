from flask import Flask, request, jsonify, render_template, send_file
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from tqdm import tqdm
import base64
from google.cloud import vision
import requests

app = Flask(__name__)

# Google Drive URL
google_drive_url = "https://drive.google.com/uc?id=17TQ7teUPE__UQT3v10J_8aRfkmKCZzLh&export=download"

# Download the file
def download_file(url, save_path):
    print(f"Downloading file from {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            file.write(response.content)
        print(f"File saved to {save_path}")
    else:
        print(f"Failed to download file, status code: {response.status_code}")

# Path to save the file
save_path = "CELEBRITY-FINDER/celebrity_features.npy"

# Download the file if it does not exist
if not os.path.exists(save_path):
    download_file(google_drive_url, save_path)
else:
    print(f"File already exists: {save_path}")

def preprocess_image(image_bytes_or_path):
    """
    Preprocess an image. Handles both file paths and raw bytes.
    """
    if isinstance(image_bytes_or_path, bytes):
        # If raw bytes are provided, decode the image
        image_array = np.frombuffer(image_bytes_or_path, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from bytes")
    else:
        # If it's a file path, read the image directly
        image = cv2.imread(image_bytes_or_path)
        if image is None:
            raise ValueError(f"Failed to load image from path: {image_bytes_or_path}")

    # Convert to RGB and resize
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = tf.image.resize(image_rgb, IMAGE_SIZE) / 255.0
    return image_resized

def extract_features_from_dataset(df, save_path="CELEBRITY-FINDER/celebrity_features.npy"):
    # Check if the feature file exists, otherwise extract features
    if os.path.exists(save_path):
        print("Loading precomputed features from file...")
        return np.load(save_path)

    # Proceed with feature extraction if no precomputed file is available
    print("Extracting features from dataset...")
    feature_vectors = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        image_dict = row["image"]  # The `image` column contains a dictionary
        image_bytes = image_dict["bytes"]  # Access the raw image bytes
        image = preprocess_image(image_bytes)
        feature_vector = feature_extractor(tf.expand_dims(image, axis=0)).numpy()
        feature_vectors.append(feature_vector.flatten())  # Flatten to 1D vector

    # Save features to file
    feature_vectors = np.array(feature_vectors)
    np.save(save_path, feature_vectors)
    print(f"Features saved to {save_path}")
    return feature_vectors

# Load model and dataset
MODEL_HANDLE = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/feature_vector/2"
IMAGE_SIZE = (224, 224)
feature_extractor = hub.KerasLayer(MODEL_HANDLE, trainable=False)
dataset = load_dataset("tonyassi/celebrity-1000", split="train")
df = dataset.to_pandas()
celebrity_features = extract_features_from_dataset(df, save_path="celebrity_features.npy")

# Google Vision API setup
google_key_path = "CELEBRITY-FINDER/google_key/strategic-arc-443523-b0-360a5e15ca65.json"
vision_client = None
if os.path.exists(google_key_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_key_path
    vision_client = vision.ImageAnnotatorClient()

@app.route("/doppelganger", methods=["POST"])
def doppelganger():
    user_image = request.files["user_image"]
    doppelganger_image = request.files["doppelganger_image"]

    # Load images
    user_img = cv2.imdecode(np.frombuffer(user_image.read(), np.uint8), cv2.IMREAD_COLOR)
    doppelganger_img = cv2.imdecode(np.frombuffer(doppelganger_image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Preprocess images
    def preprocess(img):
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized / 255.0
        return np.expand_dims(img_normalized, axis=0)

    user_features = feature_extractor(preprocess(user_img)).numpy()
    doppelganger_features = feature_extractor(preprocess(doppelganger_img)).numpy()

    # Calculate similarity
    similarity = cosine_similarity(user_features, doppelganger_features).flatten()[0] * 100

    return jsonify({"similarity_score": round(similarity, 2)})

@app.route("/basic", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    upload_dir = "./static/uploads"
    os.makedirs(upload_dir, exist_ok=True)  # Create directory if it doesn't exist
    file_path = os.path.join(upload_dir, file.filename)  # Construct file path
    file.save(file_path)

    # Process image and find matches
    input_features = feature_extractor(tf.expand_dims(preprocess_image(file_path), axis=0)).numpy()
    similarities = cosine_similarity(input_features, celebrity_features).flatten()
    top_indices = similarities.argsort()[-5:][::-1]

    # Prepare response
    matches = []
    for idx in top_indices:
        row = df.iloc[idx]
        image_bytes = row["image"]["bytes"]  # Access raw image bytes

        # Use Google Vision API if credentials are available
        description = "Unknown"
        if vision_client:
            vision_image = vision.Image(content=image_bytes)
            response = vision_client.web_detection(image=vision_image)
            if response.web_detection.web_entities:
                description = response.web_detection.web_entities[0].description

        matches.append({
            "name": str(row["label"]),  # Convert to Python string
            "similarity": float(round(similarities[idx] * 100, 2)),  # Convert to Python float
            "description": description,
            "image": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"  # Convert bytes to base64
        })

    return jsonify({"matches": matches})

# Route to render the main HTML page
@app.route("/", methods=["GET"])
def main_page():
    return send_file("index.html")

if __name__ == "__main__":
    app.run(debug=True)
