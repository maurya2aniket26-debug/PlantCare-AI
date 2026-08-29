import os
import json
import logging
import threading

from flask import Flask, render_template, request


# ============================================================
# FLASK CONFIGURATION
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

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
model_lock = threading.Lock()


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():

    global class_names

    if class_names is not None:
        return class_names

    try:

        logger.info("Loading class names...")

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            class_names = json.load(file)

        if not isinstance(class_names, list):
            raise ValueError(
                "class_names.json must contain a list."
            )

        if len(class_names) != 38:
            raise ValueError(
                f"Expected 38 classes, found {len(class_names)}."
            )

        logger.info(
            "Loaded %d PlantVillage classes.",
            len(class_names)
        )

        return class_names

    except Exception as error:

        logger.exception(
            "Could not load class_names.json: %s",
            error
        )

        raise


# ============================================================
# LOAD MODEL
# ============================================================

def get_model():

    global model

    if model is not None:
        return model

    with model_lock:

        if model is not None:
            return model

        logger.info("Loading TensorFlow...")

        import tensorflow as tf

        # CPU only - safer for Render
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

        # Limit CPU threads
        try:
            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(2)
        except Exception:
            pass

        logger.info("Loading plant disease model...")

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        logger.info("Model loaded successfully.")

        logger.info(
            "Model output shape: %s",
            model.output_shape
        )

        return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(image_file):

    import tensorflow as tf

    image_file.seek(0)

    image_bytes = image_file.read()

    if not image_bytes:
        raise ValueError(
            "The uploaded image is empty."
        )

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [224, 224]
    )

    image = tf.cast(
        image,
        tf.float32
    )

    # MobileNetV2 preprocessing
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image_file):

    import numpy as np

    current_model = get_model()
    current_classes = load_class_names()

    image = prepare_image(
        image_file
    )

    predictions = current_model.predict(
        image,
        verbose=0
    )

    predictions = np.asarray(
        predictions
    ).reshape(-1)

    if len(predictions) != len(current_classes):

        raise ValueError(
            "Model output does not match class_names.json. "
            f"Model outputs: {len(predictions)}, "
            f"classes: {len(current_classes)}"
        )

    # Make sure probabilities are valid
    total = float(
        np.sum(predictions)
    )

    if (
        np.any(predictions < 0)
        or not np.isclose(
            total,
            1.0,
            atol=0.05
        )
    ):

        exp_values = np.exp(
            predictions -
            np.max(predictions)
        )

        predictions = (
            exp_values /
            np.sum(exp_values)
        )

    # Get highest prediction
    top_indices = np.argsort(
        predictions
    )[::-1]

    best_index = int(
        top_indices[0]
    )

    second_index = int(
        top_indices[1]
    )

    confidence = float(
        predictions[best_index] * 100
    )

    second_confidence = float(
        predictions[second_index] * 100
    )

    predicted_class = current_classes[
        best_index
    ]

    margin = (
        confidence -
        second_confidence
    )

    # --------------------------------------------------------
    # UNCERTAINTY CHECK
    # --------------------------------------------------------

    if confidence < 45:

        return {
            "valid": False,
            "reason": (
                "The AI is not confident that this "
                "image is a recognizable plant leaf."
            ),
            "confidence": round(
                confidence,
                2
            )
        }

    if confidence < 60 and margin < 10:

        return {
            "valid": False,
            "reason": (
                "The image is unclear or does not "
                "look sufficiently like a PlantVillage "
                "plant image."
            ),
            "confidence": round(
                confidence,
                2
            )
        }

    return {
        "valid": True,
        "class_name": predicted_class,
        "confidence": round(
            confidence,
            2
        ),
        "margin": round(
            margin,
            2
        )
    }


# ============================================================
# SPLIT CLASS NAME
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

    plant_raw = parts[0]
    disease_raw = parts[1]

    # Plant name
    plant = plant_raw.replace(
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

    plant = plant.strip()

    # Disease name
    disease = disease_raw.replace(
        "_",
        " "
    )

    disease = disease.strip()

    return plant, disease


# ============================================================
# CARE INFORMATION
# ============================================================

CARE_INFO = {

    "Apple___Apple_scab":
        "Remove infected leaves and fallen debris. Improve air circulation and avoid prolonged leaf wetness.",

    "Apple___Black_rot":
        "Remove affected leaves and fruit. Prune infected branches with clean tools and improve ventilation.",

    "Apple___Cedar_apple_rust":
        "Remove infected leaves where practical and improve airflow. Keep the growing area clean.",

    "Apple___healthy":
        "The apple plant appears healthy. Maintain good sunlight, suitable watering and regular monitoring.",

    "Blueberry___healthy":
        "Maintain suitable soil moisture, good drainage and adequate sunlight. Remove damaged leaves.",

    "Cherry_(including_sour)___Powdery_mildew":
        "Improve air circulation and avoid excessive humidity. Remove heavily affected leaves.",

    "Cherry_(including_sour)___healthy":
        "Maintain suitable moisture, sunlight and airflow. Regularly inspect new growth.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        "Remove heavily affected leaves where practical. Improve airflow and avoid prolonged leaf wetness.",

    "Corn_(maize)___Common_rust_":
        "Monitor the leaves for rust spots. Maintain good airflow and balanced plant nutrition.",

    "Corn_(maize)___Northern_Leaf_Blight":
        "Remove heavily infected leaves and plant debris. Improve airflow and avoid prolonged leaf moisture.",

    "Corn_(maize)___healthy":
        "The corn plant appears healthy. Maintain suitable soil moisture, sunlight and balanced nutrition.",

    "Grape___Black_rot":
        "Remove infected leaves and fruit. Improve airflow and avoid prolonged leaf wetness.",

    "Grape___Esca_(Black_Measles)":
        "Remove severely affected material where appropriate. Maintain ventilation and monitor new growth.",

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        "Remove infected leaves and improve airflow. Avoid prolonged moisture on foliage.",

    "Grape___healthy":
        "Maintain good sunlight, airflow and consistent moisture. Regularly inspect leaves and fruit.",

    "Orange___Haunglongbing_(Citrus_greening)":
        "Monitor the plant closely and remove severely affected material where appropriate. Seek local agricultural guidance.",

    "Peach___Bacterial_spot":
        "Remove severely affected leaves where practical. Improve air circulation and avoid unnecessary leaf wetness.",

    "Peach___healthy":
        "Maintain adequate sunlight, suitable moisture and good airflow. Regularly inspect leaves.",

    "Pepper,_bell___Bacterial_spot":
        "Remove severely affected leaves and improve ventilation. Avoid overhead watering.",

    "Pepper,_bell___healthy":
        "Maintain consistent watering, adequate sunlight and good airflow.",

    "Potato___Early_blight":
        "Remove affected leaves and infected debris. Improve airflow and avoid prolonged leaf wetness.",

    "Potato___Late_blight":
        "Remove affected foliage promptly. Improve ventilation and avoid prolonged leaf wetness.",

    "Potato___healthy":
        "The potato plant appears healthy. Maintain suitable soil moisture, good drainage and sunlight.",

    "Raspberry___healthy":
        "Maintain good airflow, adequate sunlight and consistent moisture.",

    "Soybean___healthy":
        "Maintain balanced watering, sunlight and soil nutrition. Inspect leaves regularly.",

    "Squash___Powdery_mildew":
        "Improve airflow and avoid excessive humidity. Remove heavily affected leaves.",

    "Strawberry___Leaf_scorch":
        "Remove severely damaged leaves and improve airflow. Maintain consistent soil moisture.",

    "Strawberry___healthy":
        "Maintain suitable moisture, good sunlight and airflow. Remove damaged leaves.",

    "Tomato___Bacterial_spot":
        "Remove severely affected leaves and avoid overhead watering. Improve air circulation.",

    "Tomato___Early_blight":
        "Remove affected lower leaves and infected debris. Improve airflow and avoid prolonged leaf wetness.",

    "Tomato___Late_blight":
        "Remove affected foliage promptly. Improve airflow and avoid prolonged leaf wetness.",

    "Tomato___Leaf_Mold":
        "Improve ventilation and reduce excessive humidity. Remove heavily affected leaves.",

    "Tomato___Septoria_leaf_spot":
        "Remove infected lower leaves and debris. Improve airflow and avoid splashing water onto foliage.",

    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Inspect the underside of leaves. Improve plant hydration and remove heavily affected leaves.",

    "Tomato___Target_Spot":
        "Remove severely affected leaves and improve airflow. Avoid prolonged leaf wetness.",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Monitor the plant closely and control insect vectors such as whiteflies. Remove severely affected plants where appropriate.",

    "Tomato___Tomato_mosaic_virus":
        "Remove severely infected plants and disinfect tools after handling. Avoid spreading plant sap.",

    "Tomato___healthy":
        "The tomato plant appears healthy. Maintain suitable watering, sunlight and ventilation."
}


# ============================================================
# GET CARE
# ============================================================

def get_care(class_name):

    if class_name in CARE_INFO:
        return CARE_INFO[class_name]

    if class_name.lower().endswith(
        "___healthy"
    ):

        return (
            "The plant appears healthy. "
            "Maintain suitable sunlight, watering, "
            "air circulation and regular monitoring."
        )

    return (
        "Remove severely affected plant material, "
        "improve air circulation, avoid prolonged "
        "leaf wetness and monitor the plant closely."
    )


# ============================================================
# STATUS
# ============================================================

def get_status(disease):

    if disease.lower() == "healthy":
        return "PLANT HEALTHY"

    return "DISEASE DETECTED"


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

    if any(
        word in disease_lower
        for word in [
            "virus",
            "late blight",
            "greening"
        ]
    ):
        return "High"

    if confidence >= 80:
        return "Moderate"

    return "Mild"


# ============================================================
# PLANT LIST
# ============================================================

def get_plant_names():

    classes = load_class_names()

    plant_names = []

    for class_name in classes:

        plant, _ = split_class_name(
            class_name
        )

        if plant not in plant_names:

            plant_names.append(
                plant
            )

    return plant_names


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
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
            error=(
                "The AI system is starting. "
                "Please refresh and try again."
            )
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
        # CHECK IMAGE
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
                error="Please select a plant image."
            )

        image_file = request.files["image"]

        if image_file.filename == "":

            return render_template(
                "index.html",
                plant_names=get_plant_names(),
                plant=None,
                disease=None,
                confidence=None,
                care=None,
                status=None,
                severity=None,
                error="Please select a plant image."
            )

        # ----------------------------------------------------
        # CHECK IMAGE TYPE
        # ----------------------------------------------------

        allowed_types = {
            "image/jpeg",
            "image/png",
            "image/webp"
        }

        content_type = (
            image_file.content_type or ""
        ).lower()

        if content_type not in allowed_types:

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
                    "Please upload a JPG, PNG or WEBP image."
                )
            )

        # ----------------------------------------------------
        # RUN MODEL
        # ----------------------------------------------------

        result = predict_image(
            image_file
        )

        # ----------------------------------------------------
        # UNCERTAIN / NOT RECOGNIZED
        # ----------------------------------------------------

        if not result["valid"]:

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
                    "⚠ "
                    + result["reason"]
                    + " Please upload a clear photo "
                      "of a plant leaf."
                )
            )

        # ----------------------------------------------------
        # GET RESULT
        # ----------------------------------------------------

        class_name = result["class_name"]

        confidence = result["confidence"]

        plant, disease = split_class_name(
            class_name
        )

        care = get_care(
            class_name
        )

        status = get_status(
            disease
        )

        severity = get_severity(
            disease,
            confidence
        )

        logger.info(
            "Prediction: %s | Confidence: %.2f%%",
            class_name,
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

    except Exception as error:

        logger.exception(
            "Prediction error: %s",
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
                "⚠ Unable to analyze this image. "
                "Please upload a clear JPG, PNG or WEBP "
                "plant leaf image and try again."
            )
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return {
        "status": "ok",
        "model_file_exists": os.path.exists(
            MODEL_PATH
        ),
        "classes_file_exists": os.path.exists(
            CLASS_NAMES_PATH
        )
    }


# ============================================================
# ERROR HANDLERS
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


@app.errorhandler(500)
def internal_error(error):

    logger.exception(
        "Internal server error: %s",
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
            "The AI could not process this request. "
            "Please try again with a clear plant image."
        )
    ), 500


# ============================================================
# LOCAL DEVELOPMENT
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
