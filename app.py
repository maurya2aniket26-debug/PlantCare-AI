from flask import Flask, render_template, request
import os
import json

# ============================================================
# TENSORFLOW RESOURCE SETTINGS
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


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

# Minimum confidence required
CONFIDENCE_THRESHOLD = 65.0

# Minimum difference between top 1 and top 2
MARGIN_THRESHOLD = 10.0


# ============================================================
# LOAD CLASS NAMES
# ============================================================

print("=" * 60)
print("AI PLANT CARE SYSTEM")
print("=" * 60)

print("Base directory:")
print(BASE_DIR)

print()
print("Checking class_names.json...")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        "class_names.json not found: " +
        CLASS_NAMES_PATH
    )

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:
    class_names = json.load(f)

print(
    "Class names loaded:",
    len(class_names)
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("Loading AI model...")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found: " +
        MODEL_PATH
    )

print("Model found:")
print(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("AI model loaded successfully.")

print(
    "Model input shape:",
    model.input_shape
)

print(
    "Model output shape:",
    model.output_shape
)


# ============================================================
# MODEL / CLASS CHECK
# ============================================================

try:

    model_classes = model.output_shape[-1]

    if model_classes != len(class_names):

        raise ValueError(
            "Model output classes and class_names.json "
            "do not match."
        )

except Exception as e:

    print(
        "Model/class check error:",
        str(e)
    )


# ============================================================
# PLANT NAME
# ============================================================

def get_plant_name(class_name):

    if "___" in class_name:

        plant_name = class_name.split("___")[0]

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
    list(
        set(
            get_plant_name(name)
            for name in class_names
        )
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
            "Monitor the tree regularly."
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
            "Maintain suitable soil moisture, sunlight "
            "and good nutrition."
        )
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "care": (
            "Remove severely affected leaves and "
            "improve airflow. Avoid excessive humidity."
        )
    },

    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "care": (
            "Continue regular watering, sunlight "
            "and plant monitoring."
        )
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn",
        "disease": "Cercospora Leaf Spot / Gray Leaf Spot",
        "care": (
            "Remove severely affected plant material "
            "where practical and improve field airflow."
        )
    },

    "Corn_(maize)___Common_rust_": {
        "plant": "Corn",
        "disease": "Common Rust",
        "care": (
            "Monitor rust symptoms and maintain good "
            "plant nutrition."
        )
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "care": (
            "Remove severely affected leaves and maintain "
            "good airflow."
        )
    },

    "Corn_(maize)___healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "care": (
            "Maintain adequate water, sunlight "
            "and balanced nutrition."
        )
    },

    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "care": (
            "Remove infected berries and leaves. "
            "Improve air circulation."
        )
    },

    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca / Black Measles",
        "care": (
            "Remove severely affected plant material "
            "and maintain good vineyard sanitation."
        )
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight",
        "care": (
            "Remove affected leaves and improve airflow."
        )
    },

    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "care": (
            "Continue appropriate irrigation, sunlight "
            "and regular monitoring."
        )
    },

    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing / Citrus Greening",
        "care": (
            "Inspect regularly and control insect vectors "
            "according to local agricultural guidance."
        )
    },

    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "care": (
            "Remove severely affected material and "
            "improve airflow."
        )
    },

    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "care": (
            "Continue proper watering, sunlight "
            "and regular monitoring."
        )
    },

    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial Spot",
        "care": (
            "Remove severely infected leaves. "
            "Avoid overhead watering."
        )
    },

    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "Healthy",
        "care": (
            "Maintain good sunlight, watering "
            "and nutrition."
        )
    },

    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "care": (
            "Remove severely affected leaves and "
            "maintain good airflow."
        )
    },

    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "care": (
            "Remove affected plant material promptly "
            "and avoid prolonged leaf wetness."
        )
    },

    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "care": (
            "Continue suitable watering, sunlight "
            "and regular monitoring."
        )
    },

    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "care": (
            "Maintain good sunlight, airflow "
            "and soil moisture."
        )
    },

    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "care": (
            "Continue proper irrigation, nutrition "
            "and crop monitoring."
        )
    },

    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "care": (
            "Remove severely affected leaves and "
            "improve airflow."
        )
    },

    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "care": (
            "Remove severely damaged leaves and "
            "maintain appropriate watering."
        )
    },

    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "care": (
            "Maintain suitable moisture, sunlight "
            "and good airflow."
        )
    },

    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "care": (
            "Remove severely affected leaves. "
            "Avoid overhead watering."
        )
    },

    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "care": (
            "Remove infected lower leaves and improve "
            "airflow. Water near the soil."
        )
    },

    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "care": (
            "Remove severely affected leaves and fruit. "
            "Keep foliage dry."
        )
    },

    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "care": (
            "Improve ventilation and reduce excessive "
            "humidity."
        )
    },

    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "care": (
            "Remove affected leaves and improve airflow. "
            "Avoid splashing water onto foliage."
        )
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider Mites",
        "care": (
            "Inspect the underside of leaves and remove "
            "heavily affected leaves."
        )
    },

    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "care": (
            "Remove affected leaves and improve airflow."
        )
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "care": (
            "Inspect for whitefly activity and control "
            "insect vectors according to local guidance."
        )
    },

    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "care": (
            "Remove severely affected plants and sanitize "
            "tools after handling infected material."
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
        IMAGE_SIZE[0] -
        image.width
    ) // 2

    y = (
        IMAGE_SIZE[1] -
        image.height
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
# SIMPLE NON-PLANT IMAGE CHECK
# ============================================================

def looks_like_plant_image(image):

    image = image.convert("RGB")

    small = image.resize(
        (80, 80),
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
        (g > r * 1.08) &
        (g > b * 1.05) &
        (g > 40)
    )

    green_ratio = float(
        np.mean(green_pixels)
    )

    # --------------------------------------------------------
    # Brown / yellow leaf pixels
    # --------------------------------------------------------

    brown_pixels = (
        (r > g * 1.08) &
        (g > b * 1.08) &
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
        (g > r * 1.02) &
        (g > b * 1.02) &
        (g < 150)
    )

    dark_green_ratio = float(
        np.mean(dark_green_pixels)
    )

    print(
        "Green:",
        round(green_ratio * 100, 2),
        "%"
    )

    print(
        "Brown:",
        round(brown_ratio * 100, 2),
        "%"
    )

    print(
        "Dark green:",
        round(dark_green_ratio * 100, 2),
        "%"
    )

    # --------------------------------------------------------
    # Plant-like image
    # --------------------------------------------------------

    if green_ratio >= 0.04:
        return True

    if brown_ratio >= 0.07:
        return True

    if dark_green_ratio >= 0.04:
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

    # --------------------------------------------------------
    # FILE CHECK
    # --------------------------------------------------------

    if "image" not in request.files:

        return show_error(
            "Please select a plant image."
        )

    file = request.files["image"]

    if file.filename == "":

        return show_error(
            "Please select a plant image."
        )

    try:

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        original_image = Image.open(
            file.stream
        ).convert("RGB")

        print(
            "Image size:",
            original_image.size
        )

        # ----------------------------------------------------
        # IMAGE SIZE CHECK
        # ----------------------------------------------------

        if (
            original_image.width < 100
            or
            original_image.height < 100
        ):

            return show_error(
                "Please upload a clear plant or leaf image."
            )

        # ----------------------------------------------------
        # BASIC PLANT CHECK
        # ----------------------------------------------------

        if not looks_like_plant_image(
            original_image
        ):

            print(
                "REJECTED: image does not look like a plant"
            )

            return show_error(
                "❌ This does not appear to be a plant image. "
                "Please upload a clear leaf or plant photo."
            )

        # ----------------------------------------------------
        # PREPARE IMAGE
        # ----------------------------------------------------

        image_array = prepare_image(
            original_image
        )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )

        probabilities = np.asarray(
            predictions[0],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # CHECK OUTPUT
        # ----------------------------------------------------

        if len(probabilities) != len(class_names):

            return show_error(
                "AI model configuration error."
            )

        # ----------------------------------------------------
        # NORMALIZE IF NECESSARY
        # ----------------------------------------------------

        total = float(
            np.sum(probabilities)
        )

        if total > 0:

            probabilities = (
                probabilities / total
            )

        # ----------------------------------------------------
        # TOP PREDICTIONS
        # ----------------------------------------------------

        top_indices = np.argsort(
            probabilities
        )[-5:][::-1]

        print()
        print("TOP PREDICTIONS")

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            print(
                rank,
                class_names[int(index)],
                round(
                    float(
                        probabilities[index] * 100
                    ),
                    2
                ),
                "%"
            )

        # ----------------------------------------------------
        # FINAL PREDICTION
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = float(
            probabilities[
                predicted_index
            ] * 100
        )

        # ----------------------------------------------------
        # TOP 2
        # ----------------------------------------------------

        sorted_probabilities = np.sort(
            probabilities
        )

        top_probability = float(
            sorted_probabilities[-1]
        )

        second_probability = float(
            sorted_probabilities[-2]
        )

        margin = (
            top_probability -
            second_probability
        ) * 100

        print()
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

            print(
                "REJECTED: LOW CONFIDENCE"
            )

            return show_error(
                "❌ The AI could not identify this image "
                "reliably. Please upload a clear plant leaf image."
            )

        # ----------------------------------------------------
        # MARGIN CHECK
        # ----------------------------------------------------

        if margin < MARGIN_THRESHOLD:

            print(
                "REJECTED: UNCERTAIN"
            )

            return show_error(
                "❌ The AI is uncertain about this image. "
                "Please upload a clearer plant leaf photo."
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
            )

            care = (
                "Monitor the plant regularly and "
                "consult a local agricultural expert "
                "if symptoms continue."
            )

        # ----------------------------------------------------
        # SEVERITY
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
        # SUCCESS
        # ----------------------------------------------------

        print()
        print("IMAGE ACCEPTED")

        print(
            "Plant:",
            plant
        )

        print(
            "Disease:",
            disease
        )

        print(
            "Confidence:",
            round(confidence, 2),
            "%"
        )

        print("=" * 60)

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

        print()
        print("=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)

        print(
            repr(e)
        )

        print("=" * 60)

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
        "plants": len(plant_names),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "margin_threshold": MARGIN_THRESHOLD
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print()
    print("=" * 60)
    print("AI PLANT CARE SYSTEM")
    print("=" * 60)

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Classes:",
        len(class_names)
    )

    print(
        "Plants:",
        len(plant_names)
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
        use_reloader=False
    )
