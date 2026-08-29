import os
import json
import logging

from flask import Flask, render_template, request
from PIL import Image, ImageOps
import numpy as np


# ============================================================
# APP CONFIGURATION
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

# Maximum upload size: 10 MB
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

    if class_names is not None:
        return class_names

    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            "class_names.json not found."
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
            "plant_disease_model.keras not found."
        )

    logger.info("Loading TensorFlow...")

    import tensorflow as tf

    # CPU mode
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    logger.info("Loading plant disease model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    logger.info(
        "Model loaded successfully."
    )

    logger.info(
        "Model output shape: %s",
        model.output_shape
    )

    return model


# ============================================================
# PLANT LIST
# ============================================================

def get_plant_names():

    classes = load_class_names()

    plants = []

    for class_name in classes:

        plant = class_name.split(
            "___",
            1
        )[0]

        plant = plant.replace(
            "Corn_(maize)",
            "Corn (maize)"
        )

        plant = plant.replace(
            "Cherry_(including_sour)",
            "Cherry (including sour)"
        )

        plant = plant.replace(
            "Pepper,_bell",
            "Bell Pepper"
        )

        if plant not in plants:
            plants.append(plant)

    return plants


# ============================================================
# CLASS NAME PARSER
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

    # Plant display name
    if plant == "Corn_(maize)":
        plant_display = "Corn (maize)"

    elif plant == "Cherry_(including_sour)":
        plant_display = "Cherry (including sour)"

    elif plant == "Pepper,_bell":
        plant_display = "Bell Pepper"

    else:
        plant_display = plant

    # Disease display name
    disease_display = disease

    disease_display = disease_display.replace(
        "_",
        " "
    )

    disease_display = disease_display.replace(
        "Gray leaf spot",
        "Gray Leaf Spot"
    )

    disease_display = disease_display.replace(
        "Cercospora leaf spot Gray leaf spot",
        "Cercospora Leaf Spot / Gray Leaf Spot"
    )

    disease_display = disease_display.replace(
        "Common rust ",
        "Common Rust"
    )

    disease_display = disease_display.replace(
        "Northern Leaf Blight",
        "Northern Leaf Blight"
    )

    disease_display = disease_display.replace(
        "Black rot",
        "Black Rot"
    )

    disease_display = disease_display.replace(
        "Late blight",
        "Late Blight"
    )

    disease_display = disease_display.replace(
        "Early blight",
        "Early Blight"
    )

    disease_display = disease_display.replace(
        "healthy",
        "Healthy"
    )

    return (
        plant_display,
        disease_display.strip()
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
        "Remove infected leaves and improve airflow around the plant.",

    "Apple___healthy":
        "The apple plant appears healthy. Maintain suitable sunlight, watering and regular monitoring.",

    "Blueberry___healthy":
        "Maintain suitable soil moisture, good drainage and adequate sunlight.",

    "Cherry_(including_sour)___Powdery_mildew":
        "Improve air circulation and avoid excessive humidity. Remove heavily affected leaves.",

    "Cherry_(including_sour)___healthy":
        "Maintain suitable moisture, sunlight and good airflow.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        "Remove heavily affected leaves where practical. Improve airflow and avoid prolonged leaf wetness.",

    "Corn_(maize)___Common_rust_":
        "Monitor the leaves for rust symptoms. Maintain good airflow and balanced plant nutrition.",

    "Corn_(maize)___Northern_Leaf_Blight":
        "Remove infected leaves and plant debris. Improve airflow and avoid prolonged leaf moisture.",

    "Corn_(maize)___healthy":
        "The corn plant appears healthy. Maintain suitable soil moisture, sunlight and balanced nutrition.",

    "Grape___Black_rot":
        "Remove infected leaves and fruit. Improve airflow and avoid prolonged leaf wetness.",

    "Grape___Esca_(Black_Measles)":
        "Remove severely affected plant material and maintain good ventilation.",

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        "Remove infected leaves and improve airflow around the plant.",

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
# IMAGE VALIDATION
# ============================================================

def open_and_validate_image(file):

    file.seek(0)

    try:

        image = Image.open(file)

        # Apply phone EXIF rotation
        image = ImageOps.exif_transpose(
            image
        )

        # Verify that the image is actually readable
        image.verify()

    except Exception:

        raise ValueError(
            "Invalid image file."
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
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(image):

    # MobileNetV2 input size
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

    # Batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# SIMPLE NON-PLANT CHECK
#
# This is NOT a second AI model.
# It prevents very obvious non-leaf images from being
# automatically presented as a plant disease.
# ============================================================

def basic_plant_check(image):

    # Resize small copy
    small = image.resize(
        (160, 160),
        Image.Resampling.BILINEAR
    )

    arr = np.asarray(
        small,
        dtype=np.float32
    )

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Green vegetation-like pixels
    green_pixels = (
        (g > r * 1.05)
        & (g > b * 1.02)
        & (g > 45)
    )

    green_ratio = float(
        np.mean(green_pixels)
    )

    # Green/brown leaf-like pixels
    leaf_like = (
        (
            (g > r * 0.85)
            & (g > b * 0.85)
            & (g > 35)
        )
        |
        (
            (r > b * 1.15)
            & (g > b * 1.05)
            & (r > 45)
            & (g > 30)
        )
    )

    leaf_ratio = float(
        np.mean(leaf_like)
    )

    logger.info(
        "Basic image check: green=%.3f leaf_like=%.3f",
        green_ratio,
        leaf_ratio
    )

    # Very obvious non-plant image
    #
    # We deliberately keep this threshold conservative.
    # A yellow/brown diseased leaf should not be rejected
    # too easily.
    if (
        green_ratio < 0.002
        and leaf_ratio < 0.015
    ):

        return False

    return True


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_image(image):

    loaded_model = load_model()
    classes = load_class_names()

    input_image = prepare_image(
        image
    )

    predictions = loaded_model.predict(
        input_image,
        verbose=0
    )

    predictions = np.asarray(
        predictions
    )[0]

    # --------------------------------------------------------
    # IMPORTANT:
    # class_names.json MUST have the same order used during
    # model training.
    # --------------------------------------------------------

    if len(predictions) != len(classes):

        raise ValueError(
            "Model output classes do not match class_names.json."
        )

    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_index]
    ) * 100.0

    predicted_class = classes[
        predicted_index
    ]

    # --------------------------------------------------------
    # TOP 5 LOGGING
    # --------------------------------------------------------

    top_indices = np.argsort(
        predictions
    )[::-1][:5]

    logger.info(
        "========== TOP 5 PREDICTIONS =========="
    )

    for index in top_indices:

        logger.info(
            "%s -> %.2f%%",
            classes[index],
            float(
                predictions[index] * 100.0
            )
        )

    logger.info(
        "========================================"
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
# HOME ROUTE
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
            "Homepage error: %s",
            error
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
            error="Website could not load correctly."
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
        # FILE EXISTS?
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
                error="Please select or capture a plant image."
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
                error="Please select or capture a plant image."
            )

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        image = open_and_validate_image(
            image_file
        )

        logger.info(
            "Received image: %s",
            image_file.filename
        )

        logger.info(
            "Image size: %s",
            image.size
        )

        # ----------------------------------------------------
        # BASIC NON-PLANT CHECK
        # ----------------------------------------------------

        if not basic_plant_check(image):

            logger.warning(
                "Image rejected as obvious non-plant image."
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
                    "This does not appear to be a plant image. "
                    "Please upload a clear photo of a plant leaf."
                )
            )

        # ----------------------------------------------------
        # AI PREDICTION
        # ----------------------------------------------------

        predicted_class, confidence = predict_image(
            image
        )

        # ----------------------------------------------------
        # PARSE RESULT
        # ----------------------------------------------------

        plant, disease = parse_class_name(
            predicted_class
        )

        # ----------------------------------------------------
        # CARE
        # ----------------------------------------------------

        care = CARE_INFO.get(
            predicted_class,
            "Maintain suitable sunlight, watering and airflow. Monitor the plant regularly."
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if disease.lower() == "healthy":

            status = "PLANT HEALTHY"

        else:

            status = "DISEASE DETECTED"

        # ----------------------------------------------------
        # SEVERITY
        # ----------------------------------------------------

        severity = get_severity(
            disease,
            confidence
        )

        logger.info(
            "FINAL RESULT"
        )

        logger.info(
            "Plant: %s",
            plant
        )

        logger.info(
            "Disease: %s",
            disease
        )

        logger.info(
            "Confidence: %.2f%%",
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
            "Image validation error: %s",
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
                "Please upload a valid plant leaf image."
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
                "Please try a clear, well-lit plant leaf photo."
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
            "Please upload an image smaller than 10 MB."
        )
    ), 413


# ============================================================
# RENDER / LOCAL START
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
