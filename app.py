from flask import Flask, render_template, request
import os
import json

# ============================================================
# REDUCE CPU / RAM USAGE
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Limit uploaded image size
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plant_disease_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.json"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 50.0
MARGIN_THRESHOLD = 5.0


# ============================================================
# LOAD CLASS NAMES
# ============================================================

print("Loading class names...")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        "class_names.json not found: "
        + CLASS_NAMES_PATH
    )

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:
    class_names = json.load(f)

print("Classes loaded:", len(class_names))


# ============================================================
# PLANT NAME
# ============================================================

def get_plant_name(class_name):

    if "___" in class_name:
        name = class_name.split("___")[0]
    else:
        name = class_name

    name = name.replace(
        "_(including_sour)",
        ""
    )

    name = name.replace(
        "_(maize)",
        ""
    )

    name = name.replace(
        ",_bell",
        ""
    )

    name = name.replace(
        "_",
        " "
    )

    return name.strip()


# ============================================================
# PLANT LIST
# ============================================================

plant_names = sorted(
    set(
        get_plant_name(name)
        for name in class_names
    )
)


# ============================================================
# DISEASE INFORMATION
# ============================================================

disease_info = {

    # APPLE
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "care": "Remove infected leaves and fruit. Improve air circulation and avoid prolonged leaf wetness."
    },

    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "care": "Remove infected fruit and damaged branches. Keep the tree clean and improve airflow."
    },

    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "care": "Remove affected leaves and maintain good airflow. Monitor the tree regularly."
    },

    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "care": "The apple plant appears healthy. Continue proper watering, sunlight and regular monitoring."
    },

    # BLUEBERRY
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "care": "Maintain suitable soil moisture, sunlight and nutrition."
    },

    # CHERRY
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "care": "Remove severely affected leaves and improve airflow. Avoid excessive humidity."
    },

    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "care": "Continue regular watering, sunlight and plant monitoring."
    },

    # CORN
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn",
        "disease": "Cercospora Leaf Spot / Gray Leaf Spot",
        "care": "Remove severely affected plant material where practical. Improve airflow and avoid prolonged leaf wetness."
    },

    "Corn_(maize)___Common_rust_": {
        "plant": "Corn",
        "disease": "Common Rust",
        "care": "Monitor rust symptoms and maintain good plant nutrition. Remove heavily affected leaves when appropriate."
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "care": "Remove severely affected leaves and maintain good airflow. Avoid prolonged moisture on foliage."
    },

    "Corn_(maize)___healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "care": "Maintain adequate water, sunlight and balanced nutrition."
    },

    # GRAPE
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "care": "Remove infected berries and leaves. Improve air circulation and avoid prolonged moisture."
    },

    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca / Black Measles",
        "care": "Remove severely affected plant material and maintain good vineyard sanitation."
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight",
        "care": "Remove affected leaves and improve airflow around the vine."
    },

    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "care": "Continue appropriate irrigation, sunlight and regular monitoring."
    },

    # ORANGE
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing / Citrus Greening",
        "care": "Inspect regularly and control insect vectors according to local agricultural guidance."
    },

    # PEACH
    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "care": "Remove severely affected material and improve airflow. Avoid unnecessary leaf wetness."
    },

    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "care": "Continue proper watering, sunlight and regular monitoring."
    },

    # PEPPER
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial Spot",
        "care": "Remove severely infected leaves. Improve airflow and avoid overhead watering."
    },

    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "Healthy",
        "care": "Maintain good sunlight, watering and nutrition."
    },

    # POTATO
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "care": "Remove severely affected leaves and maintain good airflow."
    },

    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "care": "Remove affected plant material promptly and avoid prolonged leaf wetness."
    },

    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "care": "Continue suitable watering, sunlight and regular monitoring."
    },

    # RASPBERRY
    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "care": "Maintain good sunlight, airflow and soil moisture."
    },

    # SOYBEAN
    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "care": "Continue proper irrigation, nutrition and regular crop monitoring."
    },

    # SQUASH
    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "care": "Remove severely affected leaves and improve airflow. Avoid excessive humidity."
    },

    # STRAWBERRY
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "care": "Remove severely damaged leaves and maintain appropriate watering."
    },

    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "care": "Maintain suitable moisture, sunlight and good airflow."
    },

    # TOMATO
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "care": "Remove severely affected leaves. Avoid overhead watering and improve airflow."
    },

    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "care": "Remove infected lower leaves and improve airflow. Water near the soil."
    },

    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "care": "Remove severely affected leaves and fruit. Keep foliage dry."
    },

    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "care": "Improve ventilation and reduce excessive humidity. Remove severely infected leaves."
    },

    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "care": "Remove affected leaves and improve airflow. Avoid splashing water onto foliage."
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider Mites",
        "care": "Inspect the underside of leaves and remove heavily affected leaves. Monitor mite activity."
    },

    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "care": "Remove affected leaves and improve airflow. Avoid prolonged moisture."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "care": "Inspect for whitefly activity and control insect vectors according to local guidance."
    },

    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "care": "Remove severely affected plants and sanitize tools after handling infected material."
    },

    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "care": "Continue proper watering, sunlight, nutrition and regular monitoring."
    }
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 50)
print("Loading AI model...")
print("=" * 50)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file not found: "
        + MODEL_PATH
    )

try:

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully.")
    print("Model input:", model.input_shape)
    print("Model output:", model.output_shape)

except Exception as e:

    print("MODEL LOAD ERROR:")
    print(str(e))

    raise


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# ERROR PAGE
# ============================================================

def show_error(message):

    return render_template(
        "index.html",
        error=message,
        plant_names=plant_names
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        plant_names=plant_names
    )


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    print("\n" + "=" * 50)
    print("NEW IMAGE")
    print("=" * 50)

    if "image" not in request.files:

        return show_error(
            "Please select a plant image."
        )

    file = request.files["image"]

    if not file.filename:

        return show_error(
            "Please select a plant image."
        )

    try:

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        image = Image.open(
            file.stream
        ).convert("RGB")

        print(
            "Image size:",
            image.size
        )

        # ----------------------------------------------------
        # BASIC SIZE CHECK
        # ----------------------------------------------------

        if (
            image.width < 80
            or
            image.height < 80
        ):

            return show_error(
                "Please upload a larger and clearer plant image."
            )

        # ----------------------------------------------------
        # PREPARE IMAGE
        # ----------------------------------------------------

        image_array = prepare_image(
            image
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )

        probabilities = np.asarray(
            predictions[0]
        )

        # ----------------------------------------------------
        # CHECK OUTPUT
        # ----------------------------------------------------

        if len(probabilities) != len(class_names):

            print(
                "MODEL / CLASS NAME MISMATCH"
            )

            return show_error(
                "AI model configuration error. "
                "Please check class_names.json."
            )

        # ----------------------------------------------------
        # BEST PREDICTION
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index] * 100
        )

        # ----------------------------------------------------
        # SECOND BEST
        # ----------------------------------------------------

        sorted_probabilities = np.sort(
            probabilities
        )

        if len(sorted_probabilities) >= 2:

            margin = float(
                (
                    sorted_probabilities[-1]
                    -
                    sorted_probabilities[-2]
                ) * 100
            )

        else:

            margin = 100.0

        print(
            "Prediction:",
            predicted_class
        )

        print(
            "Confidence:",
            round(confidence, 2),
            "%"
        )

        print(
            "Margin:",
            round(margin, 2),
            "%"
        )

        # ----------------------------------------------------
        # CONFIDENCE CHECK
        # ----------------------------------------------------

        if confidence < CONFIDENCE_THRESHOLD:

            return show_error(
                "The AI could not identify the plant "
                "with enough confidence. "
                "Please upload a clear leaf image."
            )

        # ----------------------------------------------------
        # MARGIN CHECK
        # ----------------------------------------------------

        if margin < MARGIN_THRESHOLD:

            return show_error(
                "The AI is uncertain about this image. "
                "Please upload a clearer plant leaf image."
            )

        # ----------------------------------------------------
        # DISEASE INFORMATION
        # ----------------------------------------------------

        info = disease_info.get(
            predicted_class
        )

        if info:

            plant = info["plant"]

            disease = info["disease"]

            care = info["care"]

        else:

            plant = get_plant_name(
                predicted_class
            )

            disease = (
                predicted_class
                .split("___")[-1]
                .replace("_", " ")
            )

            care = (
                "Monitor the plant regularly. "
                "Maintain proper watering, sunlight "
                "and good airflow. Consult a local "
                "agricultural expert if symptoms continue."
            )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if disease.lower() == "healthy":

            severity = "Healthy"

            status = "HEALTHY"

        elif confidence >= 85:

            severity = "High"

            status = "DISEASE DETECTED"

        else:

            severity = "Moderate"

            status = "POSSIBLE DISEASE"

        print(
            "Plant:",
            plant
        )

        print(
            "Disease:",
            disease
        )

        print(
            "Status:",
            status
        )

        print("=" * 50)

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return render_template(
            "index.html",

            plant=plant,

            disease=disease,

            confidence=round(
                confidence,
                2
            ),

            care=care,

            severity=severity,

            status=status,

            plant_names=plant_names
        )

    except Exception as e:

        print("\nPREDICTION ERROR")
        print(str(e))

        return show_error(
            "Unable to analyze this image. "
            "Please upload a clear plant image."
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "online",
        "model": "Plant Disease Model",
        "classes": len(class_names),
        "plants": len(plant_names)
    }


# ============================================================
# RENDER START COMMAND
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print("=" * 50)
    print("AI PLANT CARE SYSTEM")
    print("=" * 50)
    print("Port:", port)
    print("Classes:", len(class_names))
    print("Plants:", len(plant_names))
    print("=" * 50)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
        threaded=False
    )
