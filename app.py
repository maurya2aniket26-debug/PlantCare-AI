from flask import Flask, render_template, request
import os
import json

# ============================================================
# TENSORFLOW SETTINGS
# MUST BE BEFORE IMPORTING TENSORFLOW
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# FLASK
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

CONFIDENCE_THRESHOLD = 55.0
MARGIN_THRESHOLD = 5.0


# ============================================================
# LOAD CLASS NAMES
# ============================================================

print("=" * 60)
print("AI PLANT CARE SYSTEM")
print("=" * 60)

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        "class_names.json not found:\n" +
        CLASS_NAMES_PATH
    )

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:
    class_names = json.load(f)

print("Classes loaded:", len(class_names))


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found:\n" +
        MODEL_PATH
    )

print("Loading AI model...")
print("Please wait...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")
print("Model input:", model.input_shape)
print("Model output:", model.output_shape)


# ============================================================
# CHECK MODEL
# ============================================================

model_classes = model.output_shape[-1]

if model_classes != len(class_names):

    raise ValueError(
        f"MODEL / CLASS NAME MISMATCH!\n"
        f"Model output classes: {model_classes}\n"
        f"JSON classes: {len(class_names)}"
    )


# ============================================================
# PLANT NAME
# ============================================================

def get_plant_name(class_name):

    if "___" in class_name:
        plant = class_name.split("___")[0]
    else:
        plant = class_name

    plant = plant.replace(
        "_(including_sour)",
        ""
    )

    plant = plant.replace(
        "_(maize)",
        ""
    )

    plant = plant.replace(
        ",_bell",
        ""
    )

    plant = plant.replace(
        "_",
        " "
    )

    plant = plant.replace(
        ",",
        ""
    )

    return plant.strip()


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
print("Plants detected by model:")

for plant in plant_names:
    print("-", plant)

print()


# ============================================================
# DISEASE INFORMATION
# ============================================================

disease_info = {

    # ---------------- APPLE ----------------

    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "care": "Remove infected leaves and fruit. Improve air circulation and avoid prolonged leaf wetness."
    },

    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "care": "Remove infected fruit and damaged branches. Keep the area clean and improve airflow."
    },

    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "care": "Remove affected leaves and monitor the plant regularly. Maintain good airflow."
    },

    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "care": "The apple plant appears healthy. Continue proper watering, sunlight and regular monitoring."
    },


    # ---------------- BLUEBERRY ----------------

    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "care": "The blueberry plant appears healthy. Maintain suitable soil moisture, sunlight and nutrition."
    },


    # ---------------- CHERRY ----------------

    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "care": "Remove severely affected leaves and improve airflow. Avoid excessive humidity around foliage."
    },

    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "care": "The cherry plant appears healthy. Continue regular watering, sunlight and monitoring."
    },


    # ---------------- CORN ----------------

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
        "care": "The corn plant appears healthy. Maintain adequate water, sunlight and balanced nutrition."
    },


    # ---------------- GRAPE ----------------

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
        "care": "The grape plant appears healthy. Continue suitable irrigation, sunlight and monitoring."
    },


    # ---------------- ORANGE ----------------

    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing / Citrus Greening",
        "care": "Inspect regularly and control insect vectors according to local agricultural guidance."
    },


    # ---------------- PEACH ----------------

    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "care": "Remove severely affected material, improve airflow and avoid unnecessary leaf wetness."
    },

    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "care": "The peach plant appears healthy. Continue proper watering, sunlight and monitoring."
    },


    # ---------------- PEPPER ----------------

    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial Spot",
        "care": "Remove severely infected leaves. Improve airflow and avoid overhead watering."
    },

    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "Healthy",
        "care": "The bell pepper plant appears healthy. Maintain sunlight, watering and nutrition."
    },


    # ---------------- POTATO ----------------

    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "care": "Remove affected leaves and maintain good airflow. Avoid prolonged moisture on foliage."
    },

    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "care": "Remove affected plant material promptly and avoid prolonged leaf wetness."
    },

    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "care": "The potato plant appears healthy. Continue suitable watering, sunlight and monitoring."
    },


    # ---------------- RASPBERRY ----------------

    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "care": "The raspberry plant appears healthy. Maintain sunlight, airflow and suitable soil moisture."
    },


    # ---------------- SOYBEAN ----------------

    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "care": "The soybean plant appears healthy. Continue proper irrigation, nutrition and crop monitoring."
    },


    # ---------------- SQUASH ----------------

    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "care": "Remove severely affected leaves and improve airflow. Avoid excessive humidity."
    },


    # ---------------- STRAWBERRY ----------------

    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "care": "Remove severely damaged leaves and maintain appropriate watering."
    },

    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "care": "The strawberry plant appears healthy. Maintain suitable moisture, sunlight and airflow."
    },


    # ---------------- TOMATO ----------------

    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "care": "Remove severely affected leaves. Avoid overhead watering and improve airflow."
    },

    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "care": "Remove infected lower leaves. Improve airflow and water near the soil."
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
        "care": "Inspect the underside of leaves. Remove heavily affected leaves and monitor mite activity."
    },

    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "care": "Remove affected leaves and improve airflow. Avoid prolonged moisture on foliage."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "care": "Monitor whitefly activity and control insect vectors according to local guidance."
    },

    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "care": "Remove severely affected plants and sanitize tools after handling infected material."
    },

    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "care": "The tomato plant appears healthy. Continue proper watering, sunlight, nutrition and monitoring."
    }
}


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")

    # IMPORTANT:
    # Resize directly to 224x224.
    # This avoids unnecessary ImageOps processing.

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

    print()
    print("=" * 60)
    print("NEW IMAGE RECEIVED")
    print("=" * 60)

    if "image" not in request.files:

        return show_error(
            "Please select or capture a plant image."
        )

    file = request.files["image"]

    if file.filename == "":

        return show_error(
            "Please select or capture a plant image."
        )

    try:

        # ====================================================
        # OPEN IMAGE
        # ====================================================

        print("Opening image...")

        image = Image.open(
            file.stream
        ).convert("RGB")

        print(
            "Image size:",
            image.size
        )


        # ====================================================
        # BASIC IMAGE CHECK
        # ====================================================

        if image.width < 80 or image.height < 80:

            return show_error(
                "Image is too small. Please upload a clearer image."
            )


        # ====================================================
        # PREPARE
        # ====================================================

        print("Preparing image...")

        image_array = prepare_image(
            image
        )

        print(
            "Prepared shape:",
            image_array.shape
        )


        # ====================================================
        # PREDICT
        # ====================================================

        print("Running AI prediction...")
        print("Please wait...")

        predictions = model.predict(
            image_array,
            verbose=0
        )

        print("Prediction completed.")


        # ====================================================
        # GET PROBABILITIES
        # ====================================================

        probabilities = np.asarray(
            predictions[0],
            dtype=np.float32
        )

        if len(probabilities) != len(class_names):

            return show_error(
                "Model configuration error. "
                "The model and class_names.json do not match."
            )


        # ====================================================
        # TOP 5
        # ====================================================

        top_indices = np.argsort(
            probabilities
        )[-5:][::-1]

        print()
        print("TOP 5 PREDICTIONS")
        print("-" * 60)

        for i, index in enumerate(
            top_indices,
            start=1
        ):

            print(
                i,
                class_names[int(index)],
                round(
                    float(probabilities[index] * 100),
                    2
                ),
                "%"
            )

        print("-" * 60)


        # ====================================================
        # FINAL PREDICTION
        # ====================================================

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


        # ====================================================
        # SECOND BEST
        # ====================================================

        sorted_probs = np.sort(
            probabilities
        )

        top_probability = float(
            sorted_probs[-1]
        )

        second_probability = float(
            sorted_probs[-2]
        )

        margin = (
            top_probability -
            second_probability
        ) * 100


        print()
        print("FINAL CLASS:")
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


        # ====================================================
        # CONFIDENCE CHECK
        # ====================================================

        if confidence < CONFIDENCE_THRESHOLD:

            print("LOW CONFIDENCE")

            return show_error(
                "The AI could not confidently identify this image. "
                "Please upload a clear close-up photo of the leaf."
            )


        # ====================================================
        # MARGIN CHECK
        # ====================================================

        if margin < MARGIN_THRESHOLD:

            print("UNCERTAIN PREDICTION")

            return show_error(
                "The AI is uncertain about this image. "
                "Please upload a clearer leaf image."
            )


        # ====================================================
        # DISEASE INFORMATION
        # ====================================================

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

            disease = predicted_class.split(
                "___"
            )[-1].replace(
                "_",
                " "
            )

            care = (
                "Monitor the plant regularly and "
                "consult a local agricultural expert "
                "if symptoms continue."
            )


        # ====================================================
        # SEVERITY
        # ====================================================

        if disease.lower() == "healthy":

            severity = "Healthy"

            status = "HEALTHY"

        elif confidence >= 85:

            severity = "High"

            status = "DISEASE DETECTED"

        else:

            severity = "Moderate"

            status = "POSSIBLE DISEASE"


        # ====================================================
        # SUCCESS
        # ====================================================

        print()
        print("RESULT")
        print("-" * 60)

        print("Plant:", plant)
        print("Disease:", disease)
        print("Confidence:", round(confidence, 2), "%")
        print("Severity:", severity)

        print("-" * 60)


        # ====================================================
        # SEND RESULT TO HTML
        # ====================================================

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

        print(type(e).__name__)
        print(str(e))

        print("=" * 60)

        return show_error(
            "Unable to analyze the image. "
            "Please try another clear plant image."
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
        "plants": len(plant_names),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "margin_threshold": MARGIN_THRESHOLD
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("AI PLANT CARE SYSTEM")
    print("=" * 60)

    print("Model:", MODEL_PATH)
    print("Classes:", len(class_names))
    print("Plants:", len(plant_names))

    print()
    print("Computer:")
    print("http://127.0.0.1:5000")

    print()
    print("Phone:")
    print("http://192.168.43.248:5000")

    print()
    print("Starting Flask...")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )
