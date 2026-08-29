import os
import json
import logging

from flask import Flask, render_template, request


# ============================================================
# APP CONFIG
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

MODEL = None
CLASS_NAMES = None


# ============================================================
# CLASS NAMES
# ============================================================

def load_class_names():

    global CLASS_NAMES

    if CLASS_NAMES is not None:
        return CLASS_NAMES

    logger.info("Loading class names...")

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        CLASS_NAMES = json.load(f)

    if len(CLASS_NAMES) != 38:

        raise ValueError(
            "class_names.json must contain exactly 38 classes."
        )

    logger.info(
        "38 classes loaded."
    )

    return CLASS_NAMES


# ============================================================
# MODEL
# ============================================================

def load_model_once():

    global MODEL

    if MODEL is not None:
        return MODEL

    logger.info("Loading TensorFlow...")

    import tensorflow as tf

    # --------------------------------------------------------
    # CPU MODE
    # --------------------------------------------------------

    try:

        tf.config.set_visible_devices(
            [],
            "GPU"
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    logger.info(
        "Loading model: %s",
        MODEL_PATH
    )

    MODEL = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    logger.info(
        "Model loaded successfully."
    )

    return MODEL


# ============================================================
# PLANT NAMES
# ============================================================

def get_plant_names():

    classes = load_class_names()

    plants = []

    for class_name in classes:

        plant_part = class_name.split(
            "___"
        )[0]

        # Make display names
        plant = plant_part

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
# SPLIT PREDICTION
# ============================================================

def split_prediction(class_name):

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

    # --------------------------------------------------------
    # PLANT NAME
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DISEASE NAME
    # --------------------------------------------------------

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
        "Monitor leaves for rust spots. Maintain good airflow and balanced plant nutrition.",

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
        "Monitor the plant closely and seek local agricultural guidance if symptoms persist.",

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
# IMAGE PREDICTION
# ============================================================

def predict_image(image_file):

    import numpy as np
    import tensorflow as tf

    model = load_model_once()
    classes = load_class_names()

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image_file.seek(0)

    image_bytes = image_file.read()

    if not image_bytes:

        raise ValueError(
            "Empty image."
        )

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    image = tf.image.resize(
        image,
        [224, 224]
    )

    # --------------------------------------------------------
    # FLOAT32
    # --------------------------------------------------------

    image = tf.cast(
        image,
        tf.float32
    )

    # --------------------------------------------------------
    # MOBILENETV2 PREPROCESSING
    # --------------------------------------------------------

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    image = tf.expand_dims(
        image,
        0
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predictions = model.predict(
        image,
        verbose=0
    )

    predictions = np.asarray(
        predictions
    ).reshape(-1)

    if len(predictions) != len(classes):

        raise ValueError(
            "Model and class_names.json do not match."
        )

    # --------------------------------------------------------
    # SOFTMAX SAFETY
    # --------------------------------------------------------

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

        predictions = tf.nn.softmax(
            predictions
        ).numpy()

    # --------------------------------------------------------
    # TOP PREDICTION
    # --------------------------------------------------------

    best_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[best_index] * 100
    )

    predicted_class = classes[
        best_index
    ]

    # --------------------------------------------------------
    # SECOND PREDICTION
    # --------------------------------------------------------

    sorted_indices = np.argsort(
        predictions
    )[::-1]

    second_index = int(
        sorted_indices[1]
    )

    second_confidence = float(
        predictions[second_index] * 100
    )

    margin = (
        confidence -
        second_confidence
    )

    return {
        "class_name": predicted_class,
        "confidence": round(
            confidence,
            2
        ),
        "second_confidence": round(
            second_confidence,
            2
        ),
        "margin": round(
            margin,
            2
        )
    }


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

        # IMPORTANT:
        # DO NOT LOAD TENSORFLOW HERE.
        # This makes the homepage much faster.

        plants = get_plant_names()

        return render_template(
            "index.html",
            plant_names=plants,
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
            error="Website could not load correctly."
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
        # IMAGE EXISTS?
        # ----------------------------------------------------

        if "image" not in request.files:

            return render_template(
                "index.html",
                plant_names=get_plant_names(),
                error="Please select a plant image.",
                plant=None,
                disease=None,
                confidence=None,
                care=None,
                status=None,
                severity=None
            )

        image_file = request.files["image"]

        if not image_file or image_file.filename == "":

            return render_template(
                "index.html",
                plant_names=get_plant_names(),
                error="Please select a plant image.",
                plant=None,
                disease=None,
                confidence=None,
                care=None,
                status=None,
                severity=None
            )

        # ----------------------------------------------------
        # FILE TYPE
        # ----------------------------------------------------

        allowed_types = {
            "image/jpeg",
            "image/jpg",
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
                error=(
                    "Please upload JPG, PNG or WEBP."
                ),
                plant=None,
                disease=None,
                confidence=None,
                care=None,
                status=None,
                severity=None
            )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        result = predict_image(
            image_file
        )

        class_name = result[
            "class_name"
        ]

        confidence = result[
            "confidence"
        ]

        margin = result[
            "margin"
        ]

        # ----------------------------------------------------
        # IMPORTANT:
        # LOW CONFIDENCE IMAGE
        # ----------------------------------------------------

        if confidence < 45:

            logger.info(
                "Low confidence image: %.2f%%",
                confidence
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
                    "⚠ The AI could not confidently "
                    "recognize this as a supported "
                    "plant leaf. Please upload a clear "
                    "leaf image."
                )
            )

        # ----------------------------------------------------
        # VERY CLOSE PREDICTIONS
        # ----------------------------------------------------

        if (
            confidence < 55
            and margin < 5
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
                error=(
                    "⚠ The image is unclear. "
                    "Please upload a clear plant leaf "
                    "photo with good lighting."
                )
            )

        # ----------------------------------------------------
        # SPLIT RESULT
        # ----------------------------------------------------

        plant, disease = split_prediction(
            class_name
        )

        # ----------------------------------------------------
        # CARE
        # ----------------------------------------------------

        care = CARE_INFO.get(
            class_name,
            (
                "Monitor the plant regularly, "
                "remove severely affected material "
                "and maintain suitable growing "
                "conditions."
            )
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
            "Prediction: %s | Plant: %s | Disease: %s | Confidence: %.2f%%",
            class_name,
            plant,
            disease,
            confidence
        )

        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

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
            "Prediction error: %s",
            e
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
                "⚠ Could not analyze this image. "
                "Please upload a clear JPG, PNG or WEBP "
                "plant leaf image."
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
        "classes_exists": os.path.exists(
            CLASS_NAMES_PATH
        )
    }


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def too_large(error):

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
