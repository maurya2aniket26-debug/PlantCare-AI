import os
import json
import logging

from flask import Flask, render_template, request
from PIL import Image, ImageOps
import numpy as np


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

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

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# GLOBALS
# ============================================================

model = None
class_names = None


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():

    global class_names

    if class_names is None:

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            class_names = json.load(f)

        logger.info(
            "Loaded %d class names",
            len(class_names)
        )

    return class_names


# ============================================================
# LOAD MODEL ONLY WHEN NEEDED
# ============================================================

def load_model():

    global model

    if model is not None:
        return model

    logger.info("Loading TensorFlow...")

    import tensorflow as tf

    # CPU only
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    logger.info("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    logger.info("Model loaded successfully.")

    return model


# ============================================================
# PLANT NAMES
# ============================================================

def get_plant_names():

    classes = load_class_names()

    plants = []

    for item in classes:

        plant = item.split(
            "___"
        )[0]

        plant = plant.replace(
            "_(maize)",
            " (maize)"
        )

        plant = plant.replace(
            "_(including_sour)",
            " (including sour)"
        )

        plant = plant.replace(
            "Pepper,_bell",
            "Bell Pepper"
        )

        plant = plant.replace(
            "_",
            " "
        )

        if plant not in plants:
            plants.append(plant)

    return plants


# ============================================================
# CLASS NAME → PLANT + DISEASE
# ============================================================

def split_class_name(class_name):

    parts = class_name.split(
        "___",
        1
    )

    if len(parts) != 2:

        return (
            class_name,
            "Unknown"
        )

    plant = parts[0]
    disease = parts[1]

    plant = plant.replace(
        "_(maize)",
        " (maize)"
    )

    plant = plant.replace(
        "_(including_sour)",
        " (including sour)"
    )

    plant = plant.replace(
        "Pepper,_bell",
        "Bell Pepper"
    )

    plant = plant.replace(
        "_",
        " "
    )

    disease = disease.replace(
        "_",
        " "
    )

    return (
        plant.strip(),
        disease.strip()
    )


# ============================================================
# CARE INFORMATION
# ============================================================

CARE_INFO = {

    "Apple___Apple_scab":
        "Remove infected leaves and fallen debris. Improve air circulation and avoid prolonged leaf wetness.",

    "Apple___Black_rot":
        "Remove affected leaves and fruit. Prune infected branches and improve ventilation.",

    "Apple___Cedar_apple_rust":
        "Remove infected leaves where practical and improve airflow.",

    "Apple___healthy":
        "The apple plant appears healthy. Maintain good sunlight, suitable watering and regular monitoring.",

    "Blueberry___healthy":
        "Maintain suitable soil moisture, good drainage and adequate sunlight.",

    "Cherry_(including_sour)___Powdery_mildew":
        "Improve air circulation and avoid excessive humidity. Remove heavily affected leaves.",

    "Cherry_(including_sour)___healthy":
        "Maintain suitable moisture, sunlight and airflow.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        "Remove heavily affected leaves where practical. Improve airflow and avoid prolonged leaf wetness.",

    "Corn_(maize)___Common_rust_":
        "Monitor leaves for rust spots. Maintain good airflow and balanced plant nutrition.",

    "Corn_(maize)___Northern_Leaf_Blight":
        "Remove infected leaves and plant debris. Improve airflow and avoid prolonged leaf moisture.",

    "Corn_(maize)___healthy":
        "The corn plant appears healthy. Maintain suitable soil moisture, sunlight and balanced nutrition.",

    "Grape___Black_rot":
        "Remove infected leaves and fruit. Improve airflow and avoid prolonged leaf wetness.",

    "Grape___Esca_(Black_Measles)":
        "Remove severely affected material and maintain good ventilation.",

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        "Remove infected leaves and improve airflow.",

    "Grape___healthy":
        "Maintain good sunlight, airflow and consistent moisture.",

    "Orange___Haunglongbing_(Citrus_greening)":
        "Monitor the plant closely and seek local agricultural guidance if symptoms persist.",

    "Peach___Bacterial_spot":
        "Remove severely affected leaves and improve air circulation.",

    "Peach___healthy":
        "Maintain adequate sunlight, suitable moisture and good airflow.",

    "Pepper,_bell___Bacterial_spot":
        "Remove severely affected leaves and improve ventilation. Avoid overhead watering.",

    "Pepper,_bell___healthy":
        "Maintain consistent watering, adequate sunlight and good airflow.",

    "Potato___Early_blight":
        "Remove affected leaves and infected debris. Improve airflow.",

    "Potato___Late_blight":
        "Remove affected foliage promptly. Improve ventilation and avoid prolonged leaf wetness.",

    "Potato___healthy":
        "The potato plant appears healthy. Maintain suitable soil moisture and sunlight.",

    "Raspberry___healthy":
        "Maintain good airflow, adequate sunlight and consistent moisture.",

    "Soybean___healthy":
        "Maintain balanced watering, sunlight and soil nutrition.",

    "Squash___Powdery_mildew":
        "Improve airflow and avoid excessive humidity. Remove heavily affected leaves.",

    "Strawberry___Leaf_scorch":
        "Remove severely damaged leaves and maintain consistent soil moisture.",

    "Strawberry___healthy":
        "Maintain suitable moisture, good sunlight and airflow.",

    "Tomato___Bacterial_spot":
        "Remove severely affected leaves and avoid overhead watering.",

    "Tomato___Early_blight":
        "Remove affected lower leaves and infected debris. Improve airflow.",

    "Tomato___Late_blight":
        "Remove affected foliage promptly and improve airflow.",

    "Tomato___Leaf_Mold":
        "Improve ventilation and reduce excessive humidity.",

    "Tomato___Septoria_leaf_spot":
        "Remove infected lower leaves and debris. Avoid splashing water onto foliage.",

    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Inspect the underside of leaves and remove heavily affected leaves.",

    "Tomato___Target_Spot":
        "Remove severely affected leaves and improve airflow.",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Monitor the plant and control insect vectors such as whiteflies.",

    "Tomato___Tomato_mosaic_virus":
        "Remove severely infected plants and disinfect tools after handling.",

    "Tomato___healthy":
        "The tomato plant appears healthy. Maintain suitable watering, sunlight and ventilation."
}


# ============================================================
# IMAGE PREPROCESSING
# IMPORTANT:
# THIS MATCHES THE PREVIOUS WORKING VERSION
# ============================================================

def prepare_image(image_file):

    image_file.seek(0)

    image = Image.open(
        image_file
    )

    # Fix phone photos that contain EXIF rotation
    image = ImageOps.exif_transpose(
        image
    )

    # Force RGB
    image = image.convert(
        "RGB"
    )

    # EXACT MODEL INPUT SIZE
    image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # MobileNetV2 preprocessing
    image_array = (
        image_array / 127.5
    ) - 1.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICT
# ============================================================

def predict_image(image_file):

    model = load_model()

    classes = load_class_names()

    image = prepare_image(
        image_file
    )

    predictions = model.predict(
        image,
        verbose=0
    )

    predictions = np.asarray(
        predictions
    )

    predictions = predictions[0]

    # --------------------------------------------------------
    # IMPORTANT:
    # DO NOT CHANGE CLASS ORDER
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_index] * 100
    )

    predicted_class = classes[
        predicted_index
    ]

    # --------------------------------------------------------
    # LOG TOP 5
    # --------------------------------------------------------

    top_indices = np.argsort(
        predictions
    )[::-1][:5]

    logger.info(
        "========== TOP PREDICTIONS =========="
    )

    for index in top_indices:

        logger.info(
            "%s -> %.2f%%",
            classes[index],
            float(predictions[index] * 100)
        )

    logger.info(
        "======================================"
    )

    return (
        predicted_class,
        round(
            confidence,
            2
        )
    )


# ============================================================
# SEVERITY
# ============================================================

def get_severity(
    disease,
    confidence
):

    if disease.lower() == "healthy":
        return "Healthy"

    disease_lower = disease.lower()

    if (
        "virus" in disease_lower
        or "late blight" in disease_lower
        or "greening" in disease_lower
    ):

        return "High"

    if confidence >= 80:
        return "Moderate"

    return "Mild"


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    try:

        return render_template(
            "index.html",
            plant_names=get_plant_names(),
            plant=None,
            disease=None,
            confidence=None,
            care=None,
            status=None,
            severity=None,
            error=None
        )

    except Exception as e:

        logger.exception(
            "Home error: %s",
            e
        )

        return render_template(
            "index.html",
            plant_names=[],
            plant=None,
            disease=None,
            confidence=None,
            care=None,
            status=None,
            severity=None,
            error="Unable to load the website."
        )


# ============================================================
# PREDICT ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # CHECK FILE
        # ----------------------------------------------------

        if "image" not in request.files:

            return render_template(
                "index.html",
                plant_names=get_plant_names(),
                plant=None,
                disease=None,
                confidence=None,
                care=None,
                status=None,
                severity=None,
                error="Please select an image."
            )

        image_file = request.files[
            "image"
        ]

        if (
            image_file is None
            or image_file.filename == ""
        ):

            return render_template(
                "index.html",
                plant_names=get_plant_names(),
                plant=None,
                disease=None,
                confidence=None,
                care=None,
                status=None,
                severity=None,
                error="Please select an image."
            )

        # ----------------------------------------------------
        # CHECK IMAGE
        # ----------------------------------------------------

        try:

            # Verify image without changing original file
            image_file.seek(0)

            test_image = Image.open(
                image_file
            )

            test_image.verify()

            image_file.seek(0)

        except Exception:

            return render_template(
                "index.html",
                plant_names=get_plant_names(),
                plant=None,
                disease=None,
                confidence=None,
                care=None,
                status=None,
                severity=None,
                error=(
                    "The selected file is not a valid image."
                )
            )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predicted_class, confidence = predict_image(
            image_file
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        plant, disease = split_class_name(
            predicted_class
        )

        care = CARE_INFO.get(
            predicted_class,
            "Maintain suitable sunlight, watering and airflow. Monitor the plant regularly."
        )

        if disease.lower() == "healthy":

            status = "PLANT HEALTHY"

        else:

            status = "DISEASE DETECTED"

        severity = get_severity(
            disease,
            confidence
        )

        logger.info(
            "FINAL RESULT: %s | %s | %.2f%%",
            plant,
            disease,
            confidence
        )

        return render_template(
            "index.html",
            plant_names=get_plant_names(),
            plant=plant,
            disease=disease,
            confidence=confidence,
            care=care,
            status=status,
            severity=severity,
            error=None
        )

    except Exception as e:

        logger.exception(
            "Prediction error"
        )

        return render_template(
            "index.html",
            plant_names=get_plant_names(),
            plant=None,
            disease=None,
            confidence=None,
            care=None,
            status=None,
            severity=None,
            error=(
                "Unable to analyze this image. "
                "Please upload a clear plant leaf image."
            )
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "ok",
        "model_exists": os.path.exists(
            MODEL_PATH
        ),
        "class_names_exists": os.path.exists(
            CLASS_NAMES_PATH
        )
    }


# ============================================================
# 413 ERROR
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        plant_names=get_plant_names(),
        plant=None,
        disease=None,
        confidence=None,
        care=None,
        status=None,
        severity=None,
        error=(
            "Image is too large. "
            "Please use an image smaller than 10 MB."
        )
    ), 413


# ============================================================
# RUN
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
        debug=False,
        threaded=True
    )
