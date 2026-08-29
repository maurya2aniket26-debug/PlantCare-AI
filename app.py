import os
import json
import logging
import threading

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image, ImageOps


# ============================================================
# FLASK CONFIGURATION
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

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}

MAX_FILE_SIZE = 10 * 1024 * 1024


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# AI VARIABLES
# ============================================================

model = None
class_names = []

model_lock = threading.Lock()
model_loading = False
model_error = None


# ============================================================
# LOAD CLASS NAMES
#
# This is small and safe to load when Flask starts.
# ============================================================

def load_class_names():

    global class_names

    if not os.path.exists(CLASS_NAMES_PATH):

        logger.error(
            "class_names.json not found: %s",
            CLASS_NAMES_PATH
        )

        return False


    try:

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)


        if isinstance(data, list):

            class_names = data


        elif isinstance(data, dict):

            # Format:
            # {"0": "Apple___Apple_scab", ...}

            try:

                keys = sorted(
                    data.keys(),
                    key=lambda x: int(x)
                )

                class_names = [
                    data[k]
                    for k in keys
                ]

            except Exception:

                if isinstance(
                    data.get("class_names"),
                    list
                ):

                    class_names = data[
                        "class_names"
                    ]

                elif isinstance(
                    data.get("classes"),
                    list
                ):

                    class_names = data[
                        "classes"
                    ]

                else:

                    class_names = list(
                        data.values()
                    )


        class_names = [
            str(x)
            for x in class_names
        ]


        logger.info(
            "Loaded %d class names.",
            len(class_names)
        )

        return True


    except Exception as e:

        logger.exception(
            "Could not load class_names.json: %s",
            e
        )

        class_names = []

        return False


# ============================================================
# LAZY MODEL LOADING
#
# IMPORTANT:
# TensorFlow and the Keras model are NOT loaded when
# the website first opens.
#
# They are loaded only when the user clicks
# "Analyze Image".
# ============================================================

def get_model():

    global model
    global model_loading
    global model_error


    # --------------------------------------------------------
    # Already loaded
    # --------------------------------------------------------

    if model is not None:

        return model


    # --------------------------------------------------------
    # Prevent two requests from loading the model
    # at the same time.
    # --------------------------------------------------------

    with model_lock:

        if model is not None:

            return model


        if model_loading:

            return None


        model_loading = True


        try:

            logger.info(
                "Loading TensorFlow..."
            )

            import tensorflow as tf


            if not os.path.exists(MODEL_PATH):

                raise FileNotFoundError(
                    "Model file not found: "
                    + MODEL_PATH
                )


            logger.info(
                "Loading Keras model..."
            )


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


            model_error = None


            return model


        except Exception as e:

            model = None

            model_error = str(e)


            logger.exception(
                "MODEL LOADING FAILED: %s",
                e
            )


            return None


        finally:

            model_loading = False


# ============================================================
# FILE CHECK
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
# IMAGE PREPARATION
# ============================================================

def prepare_image(image):

    # Correct phone photo rotation
    image = ImageOps.exif_transpose(
        image
    )


    # Convert everything to RGB
    image = image.convert(
        "RGB"
    )


    # PlantVillage / MobileNetV2 input
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


    # Add batch
    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image):

    ai_model = get_model()


    if ai_model is None:

        raise RuntimeError(
            "AI model is still loading or could not be loaded."
        )


    image_array = prepare_image(
        image
    )


    predictions = ai_model.predict(
        image_array,
        verbose=0
    )


    predictions = np.asarray(
        predictions
    )


    if predictions.ndim == 2:

        probabilities = predictions[0]

    else:

        probabilities = predictions.flatten()


    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Convert logits to probabilities if necessary
    # --------------------------------------------------------

    total = np.sum(
        probabilities
    )


    if (
        np.any(probabilities < 0)
        or
        not np.isclose(
            total,
            1.0,
            atol=0.01
        )
    ):

        import tensorflow as tf

        probabilities = tf.nn.softmax(
            probabilities
        ).numpy()


    index = int(
        np.argmax(probabilities)
    )


    confidence = float(
        probabilities[index] * 100
    )


    if index >= len(class_names):

        raise RuntimeError(
            "Model class index does not match "
            "class_names.json."
        )


    predicted_class = str(
        class_names[index]
    )


    return (
        predicted_class,
        confidence
    )


# ============================================================
# PARSE CLASS NAME
# ============================================================

def parse_class_name(class_name):

    if "___" in class_name:

        plant, disease = class_name.split(
            "___",
            1
        )

    else:

        plant = class_name
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


    if confidence >= 80:

        return "High"


    if confidence >= 55:

        return "Moderate"


    return "Mild"


# ============================================================
# CARE INFORMATION
# ============================================================

CARE = {

    "Apple|Apple scab":
        "Remove infected leaves and fallen plant material. Improve airflow and avoid prolonged leaf wetness.",

    "Apple|Black rot":
        "Remove affected leaves and fruit. Prune infected branches and keep the plant area clean.",

    "Apple|healthy":
        "Continue balanced watering, good sunlight and regular inspection of leaves and fruit.",

    "Blueberry|healthy":
        "Maintain acidic soil, consistent moisture and good sunlight. Remove damaged plant material regularly.",

    "Cherry|Powdery mildew":
        "Improve air circulation and reduce excessive humidity. Remove heavily affected leaves.",

    "Cherry|healthy":
        "Maintain good sunlight, airflow and consistent soil moisture.",

    "Corn|Cercospora leaf spot Gray leaf spot":
        "Remove severely affected leaves where practical and improve airflow. Avoid excessive leaf wetness.",

    "Corn|Common rust":
        "Monitor rust progression and remove badly affected leaves. Maintain good airflow.",

    "Corn|Northern Leaf Blight":
        "Remove heavily infected plant material and improve airflow. Avoid unnecessary leaf wetness.",

    "Corn|healthy":
        "Maintain adequate water, sunlight and balanced nutrition. Monitor leaves regularly.",

    "Grape|Black rot":
        "Remove infected leaves and fruit. Improve canopy ventilation and avoid unnecessary moisture on foliage.",

    "Grape|Esca (Black Measles)":
        "Remove severely affected plant material and maintain good vineyard sanitation.",

    "Grape|Leaf blight (Isariopsis Leaf Spot)":
        "Remove affected leaves and improve air circulation. Keep foliage as dry as practical.",

    "Grape|healthy":
        "Maintain good sunlight, canopy airflow and balanced irrigation.",

    "Peach|Bacterial spot":
        "Remove severely affected leaves and fruit. Improve airflow and avoid prolonged leaf wetness.",

    "Peach|healthy":
        "Maintain good sunlight, balanced watering and regular pruning for airflow.",

    "Pepper|Bacterial spot":
        "Remove infected leaves and avoid splashing water onto foliage. Improve airflow and sanitation.",

    "Pepper|healthy":
        "Maintain consistent moisture, sunlight and airflow.",

    "Potato|Early blight":
        "Remove severely affected leaves and maintain good airflow. Water at the soil level.",

    "Potato|Late blight":
        "Remove infected plant material promptly. Improve airflow and avoid wetting foliage.",

    "Potato|healthy":
        "Maintain consistent soil moisture, good sunlight and regular inspection.",

    "Raspberry|healthy":
        "Maintain good sunlight, airflow and consistent moisture.",

    "Soybean|healthy":
        "Maintain balanced watering and nutrition while monitoring leaves regularly.",

    "Squash|Powdery mildew":
        "Remove heavily affected leaves and improve airflow. Reduce excessive humidity.",

    "Squash|healthy":
        "Provide good sunlight, consistent soil moisture and sufficient airflow.",

    "Strawberry|Leaf scorch":
        "Remove severely affected leaves and maintain balanced watering. Improve airflow.",

    "Strawberry|healthy":
        "Maintain consistent soil moisture, sunlight and good airflow.",

    "Tomato|Bacterial spot":
        "Remove affected leaves and avoid overhead watering. Improve airflow and keep foliage dry.",

    "Tomato|Early blight":
        "Remove severely affected leaves, improve airflow and water at the soil level.",

    "Tomato|Late blight":
        "Remove infected leaves and fruit promptly. Improve airflow and avoid wetting foliage.",

    "Tomato|Leaf Mold":
        "Improve ventilation and reduce excessive humidity. Remove heavily affected leaves.",

    "Tomato|Septoria leaf spot":
        "Remove affected leaves and improve airflow. Water at the soil level.",

    "Tomato|Spider mites Two-spotted spider mite":
        "Inspect the undersides of leaves and remove heavily affected foliage. Monitor closely.",

    "Tomato|Target Spot":
        "Remove affected leaves and improve airflow. Avoid prolonged leaf wetness.",

    "Tomato|Tomato Yellow Leaf Curl Virus":
        "Remove severely infected plant material where appropriate and control insect vectors such as whiteflies.",

    "Tomato|Tomato mosaic virus":
        "Remove infected plant material and disinfect tools. Avoid transferring infection between plants.",

    "Tomato|healthy":
        "Continue balanced watering, good sunlight and airflow. Inspect leaves regularly."
}


# ============================================================
# CARE FALLBACK
# ============================================================

def get_care(
    plant,
    disease
):

    key = (
        plant
        + "|"
        + disease
    )


    if key in CARE:

        return CARE[key]


    if disease.lower() == "healthy":

        return (
            "The plant appears healthy. Continue providing "
            "suitable sunlight, balanced watering, good airflow "
            "and regular inspection."
        )


    disease_lower = disease.lower()


    if "blight" in disease_lower:

        return (
            "Remove severely affected plant material, improve "
            "airflow and avoid overhead watering. Monitor the "
            "plant regularly."
        )


    if "mildew" in disease_lower:

        return (
            "Improve ventilation and reduce excessive humidity. "
            "Remove heavily affected leaves and avoid prolonged "
            "leaf wetness."
        )


    if "rust" in disease_lower:

        return (
            "Remove heavily affected leaves and improve air "
            "circulation. Monitor the plant for progression."
        )


    if "spot" in disease_lower:

        return (
            "Remove badly affected leaves and improve airflow. "
            "Water near the soil instead of directly onto foliage."
        )


    if "rot" in disease_lower:

        return (
            "Remove affected plant material and improve ventilation. "
            "Avoid excessive moisture."
        )


    if "virus" in disease_lower:

        return (
            "Remove severely infected plant material where "
            "appropriate and disinfect gardening tools. Monitor "
            "nearby plants."
        )


    return (
        "Remove severely affected plant material where practical, "
        "maintain good airflow and avoid unnecessary leaf wetness."
    )


# ============================================================
# PLANT LIST
# ============================================================

def get_plant_names():

    plants = []


    for item in class_names:

        if "___" in item:

            plant = item.split(
                "___",
                1
            )[0]

        else:

            plant = item


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
# HOME PAGE
#
# IMPORTANT:
# This does NOT load the AI model.
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        plant_names=get_plant_names()
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    filepath = None


    try:

        # ----------------------------------------------------
        # FILE EXISTS?
        # ----------------------------------------------------

        if "image" not in request.files:

            return render_template(
                "index.html",
                error="Please select or capture a plant image.",
                plant_names=get_plant_names()
            )


        file = request.files["image"]


        if file.filename == "":

            return render_template(
                "index.html",
                error="Please select or capture a plant image.",
                plant_names=get_plant_names()
            )


        # ----------------------------------------------------
        # FILE TYPE
        # ----------------------------------------------------

        if not allowed_file(
            file.filename
        ):

            return render_template(
                "index.html",
                error="Please use JPG, PNG or WEBP image.",
                plant_names=get_plant_names()
            )


        # ----------------------------------------------------
        # OPEN IMAGE DIRECTLY FROM MEMORY
        #
        # We do NOT need to permanently save uploads.
        # This is better for Render.
        # ----------------------------------------------------

        image = Image.open(
            file.stream
        )

        image = ImageOps.exif_transpose(
            image
        )

        image = image.convert(
            "RGB"
        )

        # ----------------------------------------------------
        # PREDICT
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


        logger.info(
            "RESULT: %s | %.2f%%",
            predicted_class,
            confidence
        )


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


    except Exception as e:

        logger.exception(
            "Prediction error: %s",
            e
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
# HEALTH CHECK
#
# Does NOT load TensorFlow.
# ============================================================

@app.route(
    "/health"
)
def health():

    if model is not None:

        return {
            "status": "ok",
            "model": "loaded",
            "classes": len(class_names)
        }


    if model_loading:

        return {
            "status": "ok",
            "model": "loading",
            "classes": len(class_names)
        }


    return {
        "status": "ok",
        "model": "not_loaded_yet",
        "classes": len(class_names)
    }


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def too_large(error):

    return render_template(
        "index.html",
        error=(
            "Image is too large. "
            "Please select an image smaller than 10 MB."
        ),
        plant_names=get_plant_names()
    ), 413


# ============================================================
# GENERAL SERVER ERROR
# ============================================================

@app.errorhandler(500)
def server_error(error):

    logger.exception(
        "Internal server error."
    )


    return render_template(
        "index.html",
        error=(
            "Something went wrong. "
            "Please try again."
        ),
        plant_names=get_plant_names()
    ), 500


# ============================================================
# LOAD ONLY SMALL CLASS FILE AT STARTUP
#
# TensorFlow/model is intentionally NOT loaded here.
# ============================================================

load_class_names()


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )


    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
