import os

# ============================================================
# IMPORTANT:
# Set TensorFlow limits BEFORE importing TensorFlow.
# This helps reduce RAM/CPU usage on Render.
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import gc

from flask import Flask, render_template, request
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


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
# TENSORFLOW CPU SETTINGS
# ============================================================

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


# ============================================================
# CLASS NAMES
# ============================================================

print("=" * 60)
print("AI PLANT CARE SYSTEM")
print("=" * 60)

print("Base directory:")
print(BASE_DIR)

print()
print("Checking class_names.json...")

if not os.path.isfile(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        "class_names.json not found: " + CLASS_NAMES_PATH
    )

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:
    class_names = json.load(f)

if not isinstance(class_names, list):
    raise ValueError(
        "class_names.json must contain a list."
    )

class_names = [
    str(name).strip()
    for name in class_names
]

print("Classes loaded:", len(class_names))


# ============================================================
# MODEL
# ============================================================

print()
print("Checking model...")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found: " + MODEL_PATH
    )

print("Model found:")
print(MODEL_PATH)

print()
print("Loading AI model...")
print("This may take some time on Render...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("AI model loaded successfully.")

print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)


# ============================================================
# MODEL / CLASS CHECK
# ============================================================

MODEL_CLASS_COUNT = int(
    model.output_shape[-1]
)

JSON_CLASS_COUNT = len(class_names)

print()
print("Model classes:", MODEL_CLASS_COUNT)
print("JSON classes:", JSON_CLASS_COUNT)

if MODEL_CLASS_COUNT != JSON_CLASS_COUNT:
    raise ValueError(
        "MODEL / CLASS_NAMES MISMATCH: "
        f"model={MODEL_CLASS_COUNT}, "
        f"json={JSON_CLASS_COUNT}"
    )


# ============================================================
# PLANT NAME
# ============================================================

def get_plant_name(class_name):

    if "___" in class_name:
        plant_name = class_name.split(
            "___",
            1
        )[0]
    else:
        plant_name = class_name

    plant_name = plant_name.replace(
        "_(including_sour)",
        ""
    )

    plant_name = plant_name.replace(
        "_(maize)",
        ""
    )

    plant_name = plant_name.replace(
        ",_bell",
        ""
    )

    plant_name = plant_name.replace(
        "_",
        " "
    )

    return plant_name.strip()


# ============================================================
# PLANT LIST
# ============================================================

plant_names = sorted(
    set(
        get_plant_name(name)
        for name in class_names
    )
)

print()
print("Plants supported by model:")

for plant in plant_names:
    print("-", plant)


# ============================================================
# DISEASE INFORMATION
# ============================================================

disease_info = {

    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "care": (
            "Remove infected leaves and fruit. "
            "Improve air circulation and avoid "
            "keeping foliage wet for long periods."
        )
    },

    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "care": (
            "Remove infected fruit and damaged branches. "
            "Keep the tree clean and improve airflow."
        )
    },

    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "care": (
            "Remove affected leaves and maintain good airflow. "
            "Monitor the tree regularly during humid conditions."
        )
    },

    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "care": (
            "The apple plant appears healthy. "
            "Continue proper watering, sunlight "
            "and regular monitoring."
        )
    },

    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "care": (
            "The blueberry plant appears healthy. "
            "Maintain suitable soil moisture, "
            "sunlight and nutrition."
        )
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "care": (
            "Remove severely affected leaves and "
            "improve airflow. Avoid excessive humidity "
            "around the foliage."
        )
    },

    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "care": (
            "The cherry plant appears healthy. "
            "Continue regular watering, sunlight "
            "and plant monitoring."
        )
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn",
        "disease": "Cercospora Leaf Spot / Gray Leaf Spot",
        "care": (
            "Remove severely affected plant material "
            "where practical and improve field airflow. "
            "Avoid prolonged leaf wetness."
        )
    },

    "Corn_(maize)___Common_rust_": {
        "plant": "Corn",
        "disease": "Common Rust",
        "care": (
            "Monitor rust symptoms and maintain good "
            "plant nutrition. Remove heavily affected "
            "leaves when appropriate."
        )
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "care": (
            "Remove severely affected leaves and maintain "
            "good airflow. Avoid prolonged moisture "
            "on foliage."
        )
    },

    "Corn_(maize)___healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "care": (
            "The corn plant appears healthy. "
            "Maintain adequate water, sunlight "
            "and balanced nutrition."
        )
    },

    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "care": (
            "Remove infected berries and leaves. "
            "Improve air circulation and avoid "
            "prolonged moisture on foliage."
        )
    },

    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca / Black Measles",
        "care": (
            "Remove severely affected plant material "
            "and monitor the vine carefully. "
            "Maintain good vineyard sanitation."
        )
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight",
        "care": (
            "Remove affected leaves and improve airflow "
            "around the vine. Keep the foliage as dry "
            "as practical."
        )
    },

    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "care": (
            "The grape plant appears healthy. "
            "Continue appropriate irrigation, sunlight "
            "and regular monitoring."
        )
    },

    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing / Citrus Greening",
        "care": (
            "Inspect the plant regularly and control "
            "insect vectors according to local agricultural "
            "guidance. Consult a local plant specialist "
            "if symptoms persist."
        )
    },

    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "care": (
            "Remove severely affected material and "
            "improve airflow. Avoid unnecessary leaf "
            "wetness and monitor new growth."
        )
    },

    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "care": (
            "The peach plant appears healthy. "
            "Continue proper watering, sunlight "
            "and regular monitoring."
        )
    },

    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial Spot",
        "care": (
            "Remove severely infected leaves and "
            "improve airflow. Avoid overhead watering "
            "and keep foliage dry."
        )
    },

    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "Healthy",
        "care": (
            "The bell pepper plant appears healthy. "
            "Maintain good sunlight, watering "
            "and nutrition."
        )
    },

    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "care": (
            "Remove severely affected leaves and "
            "maintain good airflow. Avoid prolonged "
            "moisture on foliage."
        )
    },

    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "care": (
            "Remove affected plant material promptly "
            "and avoid prolonged leaf wetness. "
            "Seek local agricultural guidance if "
            "symptoms spread rapidly."
        )
    },

    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "care": (
            "The potato plant appears healthy. "
            "Continue suitable watering, sunlight "
            "and regular monitoring."
        )
    },

    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "care": (
            "The raspberry plant appears healthy. "
            "Maintain good sunlight, airflow "
            "and soil moisture."
        )
    },

    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "care": (
            "The soybean plant appears healthy. "
            "Continue proper irrigation, nutrition "
            "and regular crop monitoring."
        )
    },

    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "care": (
            "Remove severely affected leaves and "
            "improve airflow. Avoid excessive humidity "
            "and prolonged leaf wetness."
        )
    },

    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "care": (
            "Remove severely damaged leaves and "
            "maintain appropriate watering. Avoid "
            "unnecessary stress and overcrowding."
        )
    },

    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "care": (
            "The strawberry plant appears healthy. "
            "Maintain suitable moisture, sunlight "
            "and good airflow."
        )
    },

    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "care": (
            "Remove severely affected leaves and "
            "avoid overhead watering. Improve airflow "
            "and keep foliage dry."
        )
    },

    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "care": (
            "Remove infected lower leaves and improve "
            "airflow. Water near the soil rather than "
            "directly onto foliage."
        )
    },

    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "care": (
            "Remove severely affected leaves and fruit. "
            "Keep foliage dry and seek local agricultural "
            "guidance if infection spreads quickly."
        )
    },

    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "care": (
            "Improve ventilation and reduce excessive "
            "humidity. Remove severely infected leaves "
            "and avoid prolonged leaf wetness."
        )
    },

    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "care": (
            "Remove affected leaves and improve airflow. "
            "Avoid splashing water onto the foliage."
        )
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider Mites",
        "care": (
            "Inspect the underside of leaves and remove "
            "heavily affected leaves. Maintain suitable "
            "plant hydration and monitor mite activity."
        )
    },

    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "care": (
            "Remove affected leaves and improve airflow. "
            "Avoid prolonged moisture on foliage."
        )
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "care": (
            "Inspect for whitefly activity and remove "
            "severely affected plants where appropriate. "
            "Control insect vectors according to local guidance."
        )
    },

    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "care": (
            "Remove severely affected plants and sanitize "
            "tools and hands after handling infected material. "
            "Avoid spreading plant sap between plants."
        )
    },

    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "care": (
            "The tomato plant appears healthy. "
            "Continue proper watering, sunlight, "
            "nutrition and regular monitoring."
        )
    }
}


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")

    image = ImageOps.contain(
        image,
        IMAGE_SIZE,
        method=Image.Resampling.LANCZOS
    )

    background = Image.new(
        "RGB",
        IMAGE_SIZE,
        (255, 255, 255)
    )

    x = (
        IMAGE_SIZE[0] - image.width
    ) // 2

    y = (
        IMAGE_SIZE[1] - image.height
    ) // 2

    background.paste(
        image,
        (x, y)
    )

    image_array = np.asarray(
        background,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# BASIC PLANT IMAGE CHECK
# ============================================================

def looks_like_plant_image(image):

    image = image.convert("RGB")

    small = image.resize(
        (100, 100),
        Image.Resampling.LANCZOS
    )

    arr = np.asarray(
        small,
        dtype=np.float32
    )

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # --------------------------------------------------------
    # Green pixels
    # --------------------------------------------------------

    green_pixels = (
        (g > r * 1.05) &
        (g > b * 1.03) &
        (g > 45)
    )

    green_ratio = float(
        np.mean(green_pixels)
    )

    # --------------------------------------------------------
    # Brown / yellow pixels
    # --------------------------------------------------------

    brown_pixels = (
        (r > g * 1.05) &
        (g > b * 1.10) &
        (r > 50) &
        (g > 35)
    )

    brown_ratio = float(
        np.mean(brown_pixels)
    )

    # --------------------------------------------------------
    # Dark green pixels
    # --------------------------------------------------------

    dark_green_pixels = (
        (g > r * 0.95) &
        (g > b * 1.05) &
        (g > 30) &
        (r < 130)
    )

    dark_green_ratio = float(
        np.mean(dark_green_pixels)
    )

    print(
        "Green ratio:",
        round(green_ratio * 100, 2),
        "%"
    )

    print(
        "Brown ratio:",
        round(brown_ratio * 100, 2),
        "%"
    )

    print(
        "Dark-green ratio:",
        round(dark_green_ratio * 100, 2),
        "%"
    )

    # --------------------------------------------------------
    # Plant decision
    # --------------------------------------------------------

    if green_ratio >= 0.06:
        return True

    if brown_ratio >= 0.08:
        return True

    if dark_green_ratio >= 0.08:
        return True

    return False


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
# HEALTH
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
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    print()
    print("=" * 60)
    print("NEW IMAGE RECEIVED")
    print("=" * 60)

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
                "Please upload a larger and clearer "
                "plant leaf image."
            )

        # ----------------------------------------------------
        # BASIC PLANT CHECK
        # ----------------------------------------------------

        print()
        print(
            "Checking whether image looks like a plant..."
        )

        if not looks_like_plant_image(image):

            print(
                "IMAGE REJECTED - NOT A PLANT IMAGE"
            )

            return show_error(
                "This does not appear to be a plant image. "
                "Please upload a clear photo of a plant leaf."
            )

        print(
            "Image passed basic plant check."
        )

        # ----------------------------------------------------
        # PREPARE IMAGE
        # ----------------------------------------------------

        image_array = prepare_image(
            image
        )

        print(
            "Prepared image:",
            image_array.shape
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        print(
            "Running AI prediction..."
        )

        predictions = model(
            image_array,
            training=False
        ).numpy()[0]

        print(
            "Prediction completed."
        )

        # ----------------------------------------------------
        # CHECK OUTPUT
        # ----------------------------------------------------

        if len(predictions) != len(class_names):

            return show_error(
                "Model configuration error. "
                "Please check class_names.json."
            )

        # ----------------------------------------------------
        # SOFTMAX SAFETY
        # ----------------------------------------------------

        prediction_sum = float(
            np.sum(predictions)
        )

        if (
            np.any(predictions < 0)
            or
            np.any(predictions > 1)
            or
            abs(prediction_sum - 1.0) > 0.05
        ):

            predictions = tf.nn.softmax(
                predictions
            ).numpy()

        # ----------------------------------------------------
        # TOP PREDICTION
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = float(
            predictions[predicted_index] * 100
        )

        # ----------------------------------------------------
        # SECOND PREDICTION
        # ----------------------------------------------------

        sorted_values = np.sort(
            predictions
        )

        top_probability = float(
            sorted_values[-1]
        )

        second_probability = float(
            sorted_values[-2]
        )

        margin = (
            top_probability -
            second_probability
        ) * 100

        # ----------------------------------------------------
        # LOG RESULT
        # ----------------------------------------------------

        print()
        print("Prediction:")
        print(predicted_class)

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

            print(
                "Prediction rejected: low confidence."
            )

            return show_error(
                "The AI could not identify this image "
                "with enough confidence. Please upload "
                "a clear close-up photo of a plant leaf."
            )

        # ----------------------------------------------------
        # MARGIN CHECK
        # ----------------------------------------------------

        if margin < MARGIN_THRESHOLD:

            print(
                "Prediction rejected: uncertain."
            )

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

        if info is not None:

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
                .strip()
            )

            care = (
                "Monitor the plant regularly. "
                "Maintain proper watering, sunlight "
                "and airflow. Consult a local agricultural "
                "expert if symptoms continue."
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

        # ----------------------------------------------------
        # CLEAN MEMORY
        # ----------------------------------------------------

        del image_array
        del predictions

        gc.collect()

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("RESULT")
        print("=" * 60)

        print("Plant:", plant)
        print("Disease:", disease)

        print(
            "Confidence:",
            round(confidence, 2),
            "%"
        )

        print("Severity:", severity)

        print("=" * 60)

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

        print()
        print("=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)

        print(
            repr(e)
        )

        print("=" * 60)

        gc.collect()

        return show_error(
            "Unable to analyze this image. "
            "Please upload a clear plant leaf image."
        )


# ============================================================
# RUN LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "5000"
        )
    )

    print()
    print("=" * 60)
    print("SERVER STARTING")
    print("=" * 60)

    print(
        "Host: 0.0.0.0"
    )

    print(
        "Port:",
        port
    )

    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
        threaded=False
    )
