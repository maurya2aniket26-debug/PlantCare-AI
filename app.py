import os
import json
import logging

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


# ============================================================
# BASIC CONFIGURATION
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

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# GLOBAL MODEL VARIABLES
# ============================================================

model = None
class_names = []


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():

    global class_names

    if not os.path.exists(CLASS_NAMES_PATH):

        raise FileNotFoundError(
            "class_names.json was not found at: "
            + CLASS_NAMES_PATH
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)


    # --------------------------------------------------------
    # Support several common JSON formats
    # --------------------------------------------------------

    if isinstance(data, list):

        class_names = data

    elif isinstance(data, dict):

        # Format:
        # {"0": "Apple___Apple_scab", ...}

        try:

            numeric_keys = sorted(
                data.keys(),
                key=lambda x: int(x)
            )

            class_names = [
                data[key]
                for key in numeric_keys
            ]

        except (ValueError, TypeError):

            # Format:
            # {"classes": [...]}

            if "classes" in data and isinstance(
                data["classes"],
                list
            ):

                class_names = data["classes"]

            elif "class_names" in data and isinstance(
                data["class_names"],
                list
            ):

                class_names = data["class_names"]

            else:

                # Last fallback
                class_names = list(data.values())

    else:

        raise ValueError(
            "Unsupported class_names.json format."
        )


    class_names = [
        str(name)
        for name in class_names
    ]


    if len(class_names) == 0:

        raise ValueError(
            "class_names.json contains no class names."
        )


    logger.info(
        "Loaded %d class names.",
        len(class_names)
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global model

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "Model was not found at: "
            + MODEL_PATH
        )


    logger.info(
        "Loading TensorFlow/Keras model..."
    )


    # compile=False is important for deployment.
    # Prediction does not require optimizer/loss information.
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


    logger.info(
        "Model loaded successfully."
    )

    logger.info(
        "Model input shape: %s",
        model.input_shape
    )

    logger.info(
        "Model output shape: %s",
        model.output_shape
    )


# ============================================================
# INITIALIZE AI SYSTEM
# ============================================================

def initialize_ai():

    global model

    try:

        load_class_names()

        load_model()

        logger.info(
            "AI system initialized successfully."
        )

    except Exception as error:

        model = None

        logger.exception(
            "AI initialization failed: %s",
            error
        )


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(image):

    # --------------------------------------------------------
    # Correct EXIF rotation.
    # This is important for photos taken on phones.
    # --------------------------------------------------------

    image = ImageOps.exif_transpose(image)


    # --------------------------------------------------------
    # Convert all images to RGB.
    # Prevents problems with PNG transparency,
    # grayscale images, etc.
    # --------------------------------------------------------

    image = image.convert("RGB")


    # --------------------------------------------------------
    # MobileNetV2 / PlantVillage models normally use
    # 224 x 224 input.
    # --------------------------------------------------------

    image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )


    image_array = np.asarray(
        image,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # MobileNetV2 preprocessing:
    #
    # pixel 0-255
    #        ↓
    # pixel -1 to +1
    # --------------------------------------------------------

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )


    # Add batch dimension:
    #
    # (224,224,3)
    #
    # becomes
    #
    # (1,224,224,3)
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image):

    global model

    if model is None:

        raise RuntimeError(
            "AI model is not available."
        )


    processed_image = prepare_image(
        image
    )


    predictions = model.predict(
        processed_image,
        verbose=0
    )


    predictions = np.asarray(
        predictions
    )


    # --------------------------------------------------------
    # Handle normal classification output
    # --------------------------------------------------------

    if predictions.ndim == 2:

        probabilities = predictions[0]

    else:

        probabilities = predictions.flatten()


    # --------------------------------------------------------
    # If the model output is not already probabilities,
    # convert it using softmax.
    # --------------------------------------------------------

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )


    if (
        np.any(probabilities < 0)
        or
        not np.isclose(
            np.sum(probabilities),
            1.0,
            atol=0.01
        )
    ):

        probabilities = tf.nn.softmax(
            probabilities
        ).numpy()


    predicted_index = int(
        np.argmax(probabilities)
    )


    confidence = float(
        probabilities[predicted_index] * 100
    )


    # --------------------------------------------------------
    # Protect against mismatch between model classes
    # and class_names.json.
    # --------------------------------------------------------

    if predicted_index >= len(class_names):

        raise RuntimeError(
            "Model returned class index "
            + str(predicted_index)
            + " but class_names.json contains only "
            + str(len(class_names))
            + " classes."
        )


    predicted_class = str(
        class_names[predicted_index]
    )


    return predicted_class, confidence


# ============================================================
# PARSE PLANT + DISEASE
# ============================================================

def parse_prediction(predicted_class):

    # Expected PlantVillage format:
    #
    # Tomato___Late_blight
    # Apple___Apple_scab
    # Potato___healthy
    #
    # --------------------------------------------------------

    if "___" in predicted_class:

        plant, disease = predicted_class.split(
            "___",
            1
        )

    else:

        plant = predicted_class
        disease = "Unknown"


    plant = plant.replace(
        "_",
        " "
    ).strip()


    disease = disease.replace(
        "_",
        " "
    ).strip()


    return plant, disease


# ============================================================
# DISEASE STATUS
# ============================================================

def get_status(disease):

    disease_lower = disease.lower()


    if disease_lower == "healthy":

        return "PLANT HEALTHY"


    if any(
        word in disease_lower
        for word in [
            "blight",
            "rot",
            "rust",
            "mildew",
            "scab",
            "spot",
            "virus",
            "mosaic",
            "curl",
            "bacterial",
            "infected"
        ]
    ):

        return "DISEASE DETECTED"


    return "POSSIBLE DISEASE"


# ============================================================
# SEVERITY
# ============================================================

def get_severity(disease, confidence):

    disease_lower = disease.lower()


    if disease_lower == "healthy":

        return "Healthy"


    # High confidence disease
    if confidence >= 80:

        return "High"


    # Medium confidence
    if confidence >= 55:

        return "Moderate"


    return "Mild"


# ============================================================
# CARE INFORMATION
# ============================================================

CARE_INFO = {

    # --------------------------------------------------------
    # APPLE
    # --------------------------------------------------------

    "Apple|Apple scab":
        "Remove infected leaves and fallen plant material. Improve air circulation and avoid prolonged leaf wetness.",

    "Apple|Black rot":
        "Remove affected leaves and fruit. Prune infected branches and keep the plant area clean and well ventilated.",

    "Apple|healthy":
        "Continue balanced watering, good sunlight and regular inspection of leaves and fruit.",


    # --------------------------------------------------------
    # BLUEBERRY
    # --------------------------------------------------------

    "Blueberry|healthy":
        "Maintain acidic soil, consistent moisture and good sunlight. Remove damaged plant material regularly.",


    # --------------------------------------------------------
    # CHERRY
    # --------------------------------------------------------

    "Cherry|Powdery mildew":
        "Improve air circulation around the plant and remove heavily affected leaves. Avoid excessive humidity.",

    "Cherry|healthy":
        "Maintain good sunlight, airflow and consistent soil moisture. Inspect new growth regularly.",


    # --------------------------------------------------------
    # CORN
    # --------------------------------------------------------

    "Corn|Cercospora leaf spot Gray leaf spot":
        "Remove severely affected leaves where practical and improve airflow. Avoid excessive leaf wetness.",

    "Corn|Common rust":
        "Monitor rust progression and remove badly affected leaves. Maintain good airflow around plants.",

    "Corn|Northern Leaf Blight":
        "Remove heavily infected plant material and improve airflow. Avoid overhead irrigation when possible.",

    "Corn|healthy":
        "Maintain adequate water, sunlight and balanced nutrition. Monitor leaves regularly for spots or discoloration.",


    # --------------------------------------------------------
    # GRAPE
    # --------------------------------------------------------

    "Grape|Black rot":
        "Remove infected leaves and fruit. Improve canopy ventilation and avoid unnecessary moisture on foliage.",

    "Grape|Esca (Black Measles)":
        "Remove severely affected plant material and improve vineyard sanitation. Monitor the plant for progression.",

    "Grape|Leaf blight (Isariopsis Leaf Spot)":
        "Remove affected leaves and improve air circulation. Keep the foliage as dry as practical.",

    "Grape|healthy":
        "Maintain good sunlight, canopy airflow and balanced irrigation. Inspect leaves and fruit regularly.",


    # --------------------------------------------------------
    # PEACH
    # --------------------------------------------------------

    "Peach|Bacterial spot":
        "Remove severely affected leaves and fruit where practical. Improve airflow and avoid prolonged leaf wetness.",

    "Peach|healthy":
        "Maintain good sunlight, balanced watering and regular pruning for airflow.",


    # --------------------------------------------------------
    # PEPPER
    # --------------------------------------------------------

    "Pepper|Bacterial spot":
        "Remove infected leaves and avoid splashing water onto foliage. Improve airflow and sanitation.",

    "Pepper|healthy":
        "Maintain consistent moisture, sunlight and airflow. Inspect leaves regularly for spots or discoloration.",


    # --------------------------------------------------------
    # POTATO
    # --------------------------------------------------------

    "Potato|Early blight":
        "Remove severely affected leaves and maintain good airflow. Water at the soil level rather than wetting foliage.",

    "Potato|Late blight":
        "Remove infected plant material promptly and keep foliage dry. Improve airflow and monitor nearby plants.",

    "Potato|healthy":
        "Maintain consistent soil moisture, good sunlight and regular inspection for leaf spots or blight.",


    # --------------------------------------------------------
    # RASPBERRY
    # --------------------------------------------------------

    "Raspberry|healthy":
        "Maintain good sunlight, airflow and consistent moisture. Remove damaged leaves and old plant material.",


    # --------------------------------------------------------
    # SOYBEAN
    # --------------------------------------------------------

    "Soybean|healthy":
        "Maintain balanced watering and nutrition while monitoring leaves regularly for disease symptoms.",


    # --------------------------------------------------------
    # SQUASH
    # --------------------------------------------------------

    "Squash|Powdery mildew":
        "Remove heavily affected leaves and improve airflow. Keep foliage dry and avoid excessive humidity.",

    "Squash|healthy":
        "Provide good sunlight, consistent soil moisture and sufficient airflow between plants.",


    # --------------------------------------------------------
    # STRAWBERRY
    # --------------------------------------------------------

    "Strawberry|Leaf scorch":
        "Remove severely affected leaves and maintain balanced watering. Improve airflow around the plant.",

    "Strawberry|healthy":
        "Maintain consistent soil moisture, sunlight and good airflow. Remove damaged leaves regularly.",


    # --------------------------------------------------------
    # TOMATO
    # --------------------------------------------------------

    "Tomato|Bacterial spot":
        "Remove affected leaves and avoid overhead watering. Improve airflow and keep foliage as dry as possible.",

    "Tomato|Early blight":
        "Remove severely affected leaves, improve airflow and water at the soil level. Avoid prolonged leaf wetness.",

    "Tomato|Late blight":
        "Remove infected leaves and fruit promptly. Improve airflow and avoid wetting the foliage.",

    "Tomato|Leaf Mold":
        "Improve ventilation and reduce excessive humidity. Remove heavily affected leaves and avoid prolonged leaf wetness.",

    "Tomato|Septoria leaf spot":
        "Remove affected leaves and improve airflow. Water at the soil level and avoid splashing foliage.",

    "Tomato|Spider mites Two-spotted spider mite":
        "Inspect the undersides of leaves and remove heavily affected foliage. Maintain adequate plant moisture and monitor closely.",

    "Tomato|Target Spot":
        "Remove affected leaves and improve airflow. Avoid overhead irrigation and prolonged leaf wetness.",

    "Tomato|Tomato Yellow Leaf Curl Virus":
        "Remove severely infected plants or leaves where appropriate and control insect vectors such as whiteflies.",

    "Tomato|Tomato mosaic virus":
        "Remove infected plant material and disinfect tools. Avoid handling healthy plants after touching infected plants.",

    "Tomato|healthy":
        "Continue balanced watering, good sunlight and airflow. Inspect leaves regularly for early signs of disease."
}


# ============================================================
# CARE FALLBACK
# ============================================================

def get_care(plant, disease):

    # Exact match
    key = plant + "|" + disease

    if key in CARE_INFO:

        return CARE_INFO[key]


    # Healthy fallback
    if disease.lower() == "healthy":

        return (
            "The plant appears healthy. Continue providing "
            "suitable sunlight, balanced watering, good airflow "
            "and regular inspection."
        )


    # Disease-specific general fallback
    disease_lower = disease.lower()


    if "rust" in disease_lower:

        return (
            "Remove heavily affected leaves and improve "
            "air circulation. Avoid prolonged moisture on "
            "the foliage and monitor the plant regularly."
        )


    if "mildew" in disease_lower:

        return (
            "Improve ventilation and reduce excessive humidity. "
            "Remove heavily affected leaves and avoid keeping "
            "the foliage wet for long periods."
        )


    if "blight" in disease_lower:

        return (
            "Remove severely affected plant material, improve "
            "airflow and avoid overhead watering. Monitor the "
            "plant closely for further spread."
        )


    if "spot" in disease_lower:

        return (
            "Remove badly affected leaves and improve airflow. "
            "Water near the soil instead of directly onto "
            "the foliage."
        )


    if "rot" in disease_lower:

        return (
            "Remove affected plant material and improve "
            "ventilation. Avoid excessive moisture and keep "
            "the growing area clean."
        )


    if "virus" in disease_lower:

        return (
            "Remove severely infected plant material where "
            "appropriate and disinfect gardening tools. "
            "Monitor nearby plants for similar symptoms."
        )


    return (
        "Remove severely affected plant material where practical, "
        "maintain good airflow and avoid unnecessary leaf wetness. "
        "Monitor the plant regularly."
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html",
        plant_names=get_plant_names()
    )


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if model is None:

            logger.error(
                "Prediction requested but model is unavailable."
            )

            return render_template(
                "index.html",
                error=(
                    "AI model is currently unavailable. "
                    "Please try again in a moment."
                ),
                plant_names=get_plant_names()
            )


        # ----------------------------------------------------
        # Check uploaded file
        # ----------------------------------------------------

        if "image" not in request.files:

            return render_template(
                "index.html",
                error=(
                    "Please select or capture a plant image."
                ),
                plant_names=get_plant_names()
            )


        file = request.files["image"]


        if file.filename == "":

            return render_template(
                "index.html",
                error=(
                    "Please select or capture a plant image."
                ),
                plant_names=get_plant_names()
            )


        # ----------------------------------------------------
        # Validate extension
        # ----------------------------------------------------

        if not allowed_file(file.filename):

            return render_template(
                "index.html",
                error=(
                    "Unsupported image format. "
                    "Please use JPG, PNG or WEBP."
                ),
                plant_names=get_plant_names()
            )


        # ----------------------------------------------------
        # Secure filename
        # ----------------------------------------------------

        filename = secure_filename(
            file.filename
        )


        # If secure_filename produces nothing
        if not filename:

            filename = "plant_image.jpg"


        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )


        # ----------------------------------------------------
        # Save uploaded image
        # ----------------------------------------------------

        file.save(filepath)


        logger.info(
            "Image received: %s",
            filename
        )


        # ----------------------------------------------------
        # Open image safely
        # ----------------------------------------------------

        with Image.open(filepath) as image:

            # Copy into memory so the file can be safely
            # closed before prediction finishes.
            image = image.copy()


        # ----------------------------------------------------
        # AI prediction
        # ----------------------------------------------------

        predicted_class, confidence = predict_image(
            image
        )


        logger.info(
            "Prediction: %s (%.2f%%)",
            predicted_class,
            confidence
        )


        # ----------------------------------------------------
        # Parse PlantVillage class
        # ----------------------------------------------------

        plant, disease = parse_prediction(
            predicted_class
        )


        # ----------------------------------------------------
        # Additional information
        # ----------------------------------------------------

        status = get_status(
            disease
        )

        severity = get_severity(
            disease,
            confidence
        )

        care = get_care(
            plant,
            disease
        )


        # ----------------------------------------------------
        # Remove uploaded file after prediction.
        #
        # This prevents Render storage from filling up
        # after many phone/laptop uploads.
        # ----------------------------------------------------

        try:

            if os.path.exists(filepath):

                os.remove(filepath)

        except Exception as cleanup_error:

            logger.warning(
                "Could not remove uploaded file: %s",
                cleanup_error
            )


        # ----------------------------------------------------
        # Render diagnosis page
        # ----------------------------------------------------

        return render_template(
            "index.html",
            plant=plant,
            disease=disease,
            confidence=round(
                confidence,
                2
            ),
            status=status,
            severity=severity,
            care=care,
            plant_names=get_plant_names()
        )


    except tf.errors.ResourceExhaustedError:

        logger.exception(
            "TensorFlow memory error."
        )

        return render_template(
            "index.html",
            error=(
                "The server temporarily ran out of memory "
                "while processing the image. Please try "
                "again with a smaller image."
            ),
            plant_names=get_plant_names()
        )


    except Exception as error:

        logger.exception(
            "Prediction error: %s",
            error
        )

        return render_template(
            "index.html",
            error=(
                "The image could not be analyzed. "
                "Please try another clear plant image."
            ),
            plant_names=get_plant_names()
        )


# ============================================================
# GET PLANT NAMES
# ============================================================

def get_plant_names():

    plants = []


    for class_name in class_names:

        if "___" in class_name:

            plant = class_name.split(
                "___",
                1
            )[0]

        else:

            plant = class_name


        plant = plant.replace(
            "_",
            " "
        ).strip()


        if plant not in plants:

            plants.append(
                plant
            )


    return plants


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    if model is None:

        return {
            "status": "error",
            "model": "not_loaded"
        }, 503


    return {
        "status": "ok",
        "model": "loaded",
        "classes": len(class_names)
    }, 200


# ============================================================
# 413 ERROR
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        error=(
            "Image is too large. "
            "Please choose an image smaller than 10 MB."
        ),
        plant_names=get_plant_names()
    ), 413


# ============================================================
# GENERAL ERROR HANDLER
# ============================================================

@app.errorhandler(500)
def internal_error(error):

    logger.exception(
        "Internal server error."
    )

    return render_template(
        "index.html",
        error=(
            "Something went wrong while processing "
            "the request. Please try again."
        ),
        plant_names=get_plant_names()
    ), 500


# ============================================================
# STARTUP
# ============================================================

# IMPORTANT:
# We initialize the model when the application starts.
#
# Render/Gunicorn imports this file and uses "app".
# The model is therefore loaded once per worker instead
# of being loaded for every image prediction.
# ============================================================

initialize_ai()


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
        debug=False
    )
