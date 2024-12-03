from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from google.cloud import vision
from tqdm import tqdm
import base64


app = Flask(__name__)


def preprocess_image(image_path):
    # Load the image from the file path
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = tf.image.resize(image_rgb, IMAGE_SIZE) / 255.0
    return image_resized



def extract_features_from_dataset(df, save_path="celebrity_features.npy"):
    # Check if the file exists
    if os.path.exists(save_path):
        print("Loading precomputed features from file...")
        return np.load(save_path)

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
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "CELEBRITY-FINDER/google_key/strategic-arc-443523-b0-360a5e15ca65.json"
client = vision.ImageAnnotatorClient()


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
        vision_image = vision.Image(content=image_bytes)
        response = client.web_detection(image=vision_image)

        # Use Google Vision API to find a relevant URL or name
        if response.web_detection.web_entities:
            description = response.web_detection.web_entities[0].description
        else:
            description = "Unknown"
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
    return render_template("main.html")

if __name__ == "__main__":
    app.run(debug=True)

