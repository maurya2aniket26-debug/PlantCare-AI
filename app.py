import os
import json
import logging

from flask import Flask, render_template, request
from PIL import Image, ImageOps
import numpy as np


# ============================================================
# FLASK APP
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
# GLOBAL VARIABLES
# ============================================================

model = None
class_names = None


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():

    global class_names

    if class_names is not None:
        return class_names

    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            "class_names.json was not found."
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(file)

    logger.info(
        "Loaded %d classes.",
        len(class_names)
    )

    return class_names


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global model

    if model is not None:
        return model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "plant_disease_model.keras was not found."
        )

    logger.info("Loading TensorFlow...")

    import tensorflow as tf

    # Force CPU
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    logger.info("Loading AI model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    logger.info(
        "AI model loaded successfully."
    )

    logger.info(
        "Model input shape: %s",
        model.input_shape
    )

    logger.info(
        "Model output shape: %s",
        model.output_shape
    )

    return model


# ============================================================
# DISPLAY PLANT NAMES
# ============================================================

def get_plant_names():

    classes = load_class_names()

    plants = []

    for class_name in classes:

        plant = class_name.split(
            "___",
            1
        )[0]

        if plant == "Corn_(maize)":
            plant = "Corn (maize)"

        elif plant == "Cherry_(including_sour)":
            plant = "Cherry (including sour)"

        elif plant == "Pepper,_bell":
            plant = "Bell Pepper"

        if plant not in plants:
            plants.append(plant)

    return plants


# ============================================================
# CONVERT MODEL CLASS TO DISPLAY TEXT
# ============================================================

def parse_class_name(class_name):

    if "___" not in class_name:

        return (
            class_name,
            "Unknown"
        )

    plant, disease = class_name.split(
        "___",
        1
    )

    # Plant name
    if plant == "Corn_(maize)":
        plant_display = "Corn (maize)"

    elif plant == "Cherry_(including_sour)":
        plant_display = "Cherry (including sour)"

    elif plant == "Pepper,_bell":
        plant_display = "Bell Pepper"

    else:
        plant_display = plant

    # Disease name
    disease_display = disease.replace(
        "_",
        " "
    )

    disease_display = disease_display.replace(
        "  ",
        " "
    )

    if disease_display.lower() == "healthy":
        disease_display = "Healthy"

    return (
        plant_display.strip(),
        disease_display.strip()
    )


# ============================================================
# CARE INFORMATION
# ============================================================

CARE_INFO = {

    "Apple___Apple_scab":
        "Remove infected leaves and fallen debris. Improve air circulation and avoid keeping leaves wet for long periods.",

    "Apple___Black_rot":
        "Remove affected leaves and fruit. Prune infected branches and improve ventilation.",

    "Apple___Cedar_apple_rust":
        "Remove infected leaves where possible and maintain good airflow.",

    "Apple___healthy":
        "The apple plant appears healthy. Maintain suitable sunlight, watering, drainage and regular monitoring.",

    "Blueberry___healthy":
        "Maintain suitable soil moisture, good drainage, sunlight and airflow.",

    "Cherry_(including_sour)___Powdery_mildew":
        "Improve air circulation and reduce excessive humidity. Remove heavily affected leaves.",

    "Cherry_(including_sour)___healthy":
        "Maintain suitable moisture, sunlight, drainage and good airflow.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        "Remove heavily affected leaves where practical. Improve airflow and avoid prolonged leaf wetness.",

    "Corn_(maize)___Common_rust_":
        "Monitor leaves for rust symptoms. Maintain good airflow and balanced plant nutrition.",

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
        "Monitor the plant carefully and seek local agricultural guidance if symptoms continue.",

    "Peach___Bacterial_spot":
        "Remove severely affected leaves and improve air circulation.",

    "Peach___healthy":
        "Maintain adequate sunlight, suitable moisture and good airflow.",

    "Pepper,_bell___Bacterial_spot":
        "Remove severely affected leaves and avoid overhead watering. Improve ventilation.",

    "Pepper,_bell___healthy":
        "Maintain consistent watering, adequate sunlight and good airflow.",

    "Potato___Early_blight":
        "Remove affected leaves and infected debris. Improve airflow around the plants.",

    "Potato___Late_blight":
        "Remove affected foliage promptly. Improve ventilation and avoid prolonged leaf wetness.",

    "Potato___healthy":
        "The potato plant appears healthy. Maintain suitable soil moisture and sunlight.",

    "Raspberry___healthy":
        "Maintain good airflow, adequate sunlight and consistent moisture.",

    "Soybean___healthy":
        "Maintain balanced watering, sunlight and suitable soil nutrition.",

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
        "Improve ventilation and reduce excessive humidity around the foliage.",

    "Tomato___Septoria_leaf_spot":
        "Remove infected lower leaves and debris. Avoid splashing water onto foliage.",

    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Inspect the underside of leaves and remove heavily affected leaves.",

    "Tomato___Target_Spot":
        "Remove severely affected leaves and improve airflow.",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Monitor the plant carefully and control insect vectors such as whiteflies.",

    "Tomato___Tomato_mosaic_virus":
        "Remove severely infected plants and disinfect tools after handling.",

    "Tomato___healthy":
        "The tomato plant appears healthy. Maintain suitable watering, sunlight and ventilation."
}


# ============================================================
# IMAGE OPENING
# ============================================================

def open_image(file):

    file.seek(0)

    try:

        image = Image.open(file)

        image = ImageOps.exif_transpose(
            image
        )

        image.verify()

    except Exception as error:

        logger.warning(
            "Invalid image: %s",
            error
        )

        raise ValueError(
            "Invalid image."
        )

    file.seek(0)

    image = Image.open(file)

    image = ImageOps.exif_transpose(
        image
    )

    image = image.convert(
        "RGB"
    )

    return image


# ============================================================
# PREPROCESSING
#
# MobileNetV2:
# 224 x 224
# RGB
# pixel range -> [-1, 1]
# ============================================================

def preprocess_image(image):

    image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array = (
        array / 127.5
    ) - 1.0

    array = np.expand_dims(
        array,
        axis=0
    )

    return array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image):

    ai_model = load_model()
    classes = load_class_names()

    processed = preprocess_image(
        image
    )

    prediction = ai_model.predict(
        processed,
        verbose=0
    )

    prediction = np.asarray(
        prediction
    )[0]

    # --------------------------------------------------------
    # IMPORTANT SAFETY CHECK
    # --------------------------------------------------------

    if len(prediction) != len(classes):

        logger.error(
            "Model classes: %d | JSON classes: %d",
            len(prediction),
            len(classes)
        )

        raise ValueError(
            "Model output does not match class_names.json."
        )

    # --------------------------------------------------------
    # GET HIGHEST PROBABILITY
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[predicted_index]
    ) * 100

    predicted_class = classes[
        predicted_index
    ]

    # --------------------------------------------------------
    # TOP 5 LOG
    # --------------------------------------------------------

    top_indices = np.argsort(
        prediction
    )[::-1][:5]

    logger.info(
        "========== AI TOP 5 =========="
    )

    for index in top_indices:

        logger.info(
            "%s -> %.2f%%",
            classes[index],
            float(
                prediction[index] * 100
            )
        )

    logger.info(
        "==============================="
    )

    return (
        predicted_class,
        round(confidence, 2)
    )


# ============================================================
# SEVERITY
# ============================================================

def get_severity(
    disease,
    confidence
):

    disease_lower = disease.lower()

    if disease_lower == "healthy":

        return "Healthy"

    if (
        "late blight" in disease_lower
        or "virus" in disease_lower
        or "greening" in disease_lower
        or "esca" in disease_lower
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

    except Exception as error:

        logger.exception(
            "Homepage error"
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
            error="Website could not be loaded."
        )


# ============================================================
# PREDICT
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
                error="Please select an image first."
            )

        uploaded_file = request.files[
            "image"
        ]

        if (
            uploaded_file is None
            or uploaded_file.filename == ""
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
                error="Please select an image first."
            )

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        image = open_image(
            uploaded_file
        )

        logger.info(
            "Received image: %s",
            uploaded_file.filename
        )

        logger.info(
            "Image size: %s",
            image.size
        )

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        predicted_class, confidence = predict_image(
            image
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        plant, disease = parse_class_name(
            predicted_class
        )

        care = CARE_INFO.get(
            predicted_class,
            "Maintain suitable sunlight, watering, drainage and airflow. Monitor the plant regularly."
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
            "FINAL PREDICTION: %s | %s | %.2f%%",
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

    except ValueError as error:

        logger.warning(
            "Prediction validation error: %s",
            error
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
                "Please upload a valid plant image."
            )
        )

    except Exception as error:

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
                "Unable to analyze the image. "
                "Please try again with a clear plant leaf photo."
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
# FILE TOO LARGE
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
# START
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
