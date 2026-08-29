import os
import base64
import uuid

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from PIL import Image


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plant_disease_mobilenet_finetuned.keras"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# 38 PLANTVILLAGE CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",

    "Blueberry___healthy",

    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",

    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",

    "Orange___Haunglongbing_(Citrus_greening)",

    "Peach___Bacterial_spot",
    "Peach___healthy",

    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",

    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",

    "Raspberry___healthy",

    "Soybean___healthy",

    "Squash___Powdery_mildew",

    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",

    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "care": "Remove infected leaves and fallen plant debris. Improve air circulation and avoid watering the leaves."
    },

    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "care": "Remove infected fruits and branches. Prune affected areas and keep the tree area clean."
    },

    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "care": "Remove infected leaves and improve air circulation. Avoid excessive moisture around the foliage."
    },

    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "care": "The apple leaf appears healthy. Continue proper watering, sunlight, nutrition and regular monitoring."
    },

    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "care": "The blueberry leaf appears healthy. Maintain suitable soil moisture, sunlight and plant nutrition."
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "care": "Improve air circulation and reduce humidity around the leaves. Remove severely affected plant material."
    },

    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "care": "The cherry leaf appears healthy. Continue regular watering, sunlight and plant monitoring."
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn",
        "disease": "Cercospora Leaf Spot / Gray Leaf Spot",
        "care": "Remove severely affected leaves and improve field air circulation. Avoid prolonged leaf wetness."
    },

    "Corn_(maize)___Common_rust_": {
        "plant": "Corn",
        "disease": "Common Rust",
        "care": "Monitor the crop regularly. Remove heavily affected leaves where practical and maintain good crop management."
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "care": "Remove infected crop debris and improve field management. Monitor nearby plants for new symptoms."
    },

    "Corn_(maize)___healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "care": "The corn leaf appears healthy. Continue adequate water, nutrients and regular crop monitoring."
    },

    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "care": "Remove infected leaves and fruit. Keep the vineyard clean and improve air circulation around plants."
    },

    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca / Black Measles",
        "care": "Remove severely affected plant material and monitor vines carefully. Maintain good vineyard sanitation."
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight",
        "care": "Remove infected leaves and improve ventilation. Avoid excessive moisture remaining on foliage."
    },

    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "care": "The grape leaf appears healthy. Maintain good sunlight, watering and vineyard hygiene."
    },

    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing / Citrus Greening",
        "care": "Monitor the plant for yellowing and uneven leaf symptoms. Manage insect vectors and consult a local agricultural expert."
    },

    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "care": "Remove severely infected leaves and maintain good airflow. Avoid unnecessary wetting of foliage."
    },

    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "care": "The peach leaf appears healthy. Continue regular watering, sunlight and monitoring."
    },

    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial Spot",
        "care": "Remove affected leaves and avoid overhead watering. Keep plants spaced to improve air circulation."
    },

    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "Healthy",
        "care": "The pepper leaf appears healthy. Maintain proper watering, sunlight and balanced nutrition."
    },

    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "care": "Remove infected leaves and plant debris. Avoid prolonged leaf wetness and maintain good crop spacing."
    },

    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "care": "Remove severely infected plant material and avoid overhead irrigation. Monitor nearby plants because the disease can spread quickly."
    },

    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "care": "The potato leaf appears healthy. Continue appropriate watering, sunlight and regular crop inspection."
    },

    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "care": "The raspberry leaf appears healthy. Maintain good soil moisture, sunlight and plant nutrition."
    },

    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "care": "The soybean leaf appears healthy. Continue proper irrigation, sunlight and regular monitoring."
    },

    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "care": "Improve air circulation and avoid excessive humidity. Remove severely affected leaves and monitor plant spread."
    },

    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "care": "Remove severely damaged leaves and maintain suitable watering. Avoid unnecessary stress and monitor new growth."
    },

    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "care": "The strawberry leaf appears healthy. Maintain adequate water, sunlight and balanced nutrition."
    },

    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "care": "Remove affected leaves and avoid overhead watering. Improve airflow and keep the growing area clean."
    },

    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "care": "Remove infected lower leaves and plant debris. Water at the soil level and improve air circulation."
    },

    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "care": "Remove severely infected leaves and fruit. Avoid overhead watering and isolate affected plants when possible."
    },

    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "care": "Reduce humidity and improve ventilation. Avoid wetting the foliage and remove severely affected leaves."
    },

    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "care": "Remove infected leaves and plant debris. Water near the soil and maintain good airflow."
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider Mites",
        "care": "Inspect the undersides of leaves. Remove heavily affected leaves and monitor plants closely for mite activity."
    },

    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "care": "Remove infected leaves and improve air circulation. Avoid prolonged leaf moisture and keep the area clean."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "care": "Monitor for whitefly activity and remove severely affected plants where appropriate. Control insect vectors and maintain plant hygiene."
    },

    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "care": "Remove severely infected plants and sanitize tools. Avoid handling healthy plants after touching infected material."
    },

    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "care": "The tomato leaf appears healthy. Continue proper watering, sunlight, nutrition and regular disease monitoring."
    }
}


# ============================================================
# LOAD MODEL
# ============================================================

print("========================================")
print("LOADING PLANT DISEASE MODEL")
print("========================================")

print("Model path:")
print(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((160, 160))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = image_array / 255.0

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    top_indices = np.argsort(
        predictions
    )[-3:][::-1]

    results = []

    for index in top_indices:

        class_name = CLASS_NAMES[index]

        confidence = float(
            predictions[index] * 100
        )

        results.append({
            "class_name": class_name,
            "confidence": confidence
        })

    return results


# ============================================================
# SAVE CAMERA IMAGE
# ============================================================

def save_camera_image(camera_data):

    if "," in camera_data:

        camera_data = camera_data.split(
            ",",
            1
        )[1]

    image_bytes = base64.b64decode(
        camera_data
    )

    filename = (
        "camera_"
        + uuid.uuid4().hex
        + ".jpg"
    )

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    with open(
        file_path,
        "wb"
    ) as file:

        file.write(image_bytes)

    return file_path, filename


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    plant = None
    confidence = None
    solution = None
    confidence_message = None
    prediction_status = None
    image_url = None
    error = None

    if request.method == "POST":

        try:

            file = request.files.get(
                "image"
            )

            camera_image_data = request.form.get(
                "camera_image_data",
                ""
            )

            file_path = None
            filename = None


            # ==================================================
            # CAMERA IMAGE
            # ==================================================

            if camera_image_data:

                file_path, filename = save_camera_image(
                    camera_image_data
                )


            # ==================================================
            # NORMAL FILE UPLOAD
            # ==================================================

            elif file and file.filename:

                original_name = os.path.basename(
                    file.filename
                )

                extension = os.path.splitext(
                    original_name
                )[1].lower()

                allowed_extensions = [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp"
                ]

                if extension not in allowed_extensions:

                    error = (
                        "Please upload a JPG, JPEG, PNG or WEBP image."
                    )

                    return render_template(
                        "index.html",
                        prediction=prediction,
                        plant=plant,
                        confidence=confidence,
                        solution=solution,
                        confidence_message=confidence_message,
                        prediction_status=prediction_status,
                        image_url=image_url,
                        error=error
                    )


                filename = (
                    uuid.uuid4().hex
                    + extension
                )

                file_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    filename
                )

                file.save(
                    file_path
                )


            # ==================================================
            # NO IMAGE
            # ==================================================

            else:

                error = (
                    "Please choose an image or take a photo."
                )

                return render_template(
                    "index.html",
                    prediction=prediction,
                    plant=plant,
                    confidence=confidence,
                    solution=solution,
                    confidence_message=confidence_message,
                    prediction_status=prediction_status,
                    image_url=image_url,
                    error=error
                )


            # ==================================================
            # PREDICTION
            # ==================================================

            results = predict_image(
                file_path
            )

            best = results[0]

            class_name = best["class_name"]

            confidence = round(
                best["confidence"],
                2
            )

            info = DISEASE_INFO.get(
                class_name,
                {
                    "plant": "Unknown",
                    "disease": class_name,
                    "care": "Please consult an agricultural expert for further assessment."
                }
            )

            plant = info["plant"]

            prediction = info["disease"]

            solution = info["care"]


            # ==================================================
            # CONFIDENCE HANDLING
            # ==================================================

            if confidence < 55:

                prediction_status = "uncertain"

                confidence_message = (
                    "The AI confidence is low. "
                    "Try uploading a clear photo of one leaf, "
                    "with good lighting and minimal background."
                )

                prediction = "Unable to Confirm"

                plant = "Unknown"

                solution = (
                    "The model is not sufficiently confident "
                    "to provide a reliable plant disease identification."
                )

            else:

                prediction_status = "normal"

                if confidence >= 85:

                    confidence_message = (
                        "The AI has high confidence in this prediction."
                    )

                elif confidence >= 70:

                    confidence_message = (
                        "The AI has moderate-to-high confidence "
                        "in this prediction."
                    )

                else:

                    confidence_message = (
                        "The AI has moderate confidence. "
                        "For better accuracy, try a clearer leaf image."
                    )


            image_url = (
                "/static/uploads/"
                + filename
            )


        except Exception as e:

            print(
                "ERROR:",
                str(e)
            )

            error = (
                "Unable to process the image. "
                "Please try another clear plant leaf image."
            )


    return render_template(
        "index.html",
        prediction=prediction,
        plant=plant,
        confidence=confidence,
        solution=solution,
        confidence_message=confidence_message,
        prediction_status=prediction_status,
        image_url=image_url,
        error=error
    )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
