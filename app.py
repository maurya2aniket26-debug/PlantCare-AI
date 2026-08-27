from flask import Flask, render_template, request
import os
import json
import gc

# ============================================================
# RENDER / TENSORFLOW MEMORY SETTINGS
# IMPORTANT:
# These MUST be set BEFORE importing TensorFlow.
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


# ============================================================
# TENSORFLOW CPU THREAD LIMIT
# ============================================================

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


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

CONFIDENCE_THRESHOLD = 65.0
MARGIN_THRESHOLD = 10.0

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("Loading Plant Disease AI model...")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found:\n" + MODEL_PATH
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")
print("Model input shape:", model.input_shape)

# Render memory cleanup after model loading
gc.collect()


# ============================================================
# LOAD CLASS NAMES
# ============================================================

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        "Class names file not found:\n" +
        CLASS_NAMES_PATH
    )

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:

    class_names = json.load(f)


print("Number of classes:", len(class_names))


# ============================================================
# CHECK MODEL / CLASS COUNT
# ============================================================

try:

    model_output_classes = model.output_shape[-1]

    print(
        "Model output classes:",
        model_output_classes
    )

    print(
        "JSON classes:",
        len(class_names)
    )

    if model_output_classes != len(class_names):

        print()
        print("WARNING!")
        print(
            "Model classes and JSON classes do not match."
        )
        print(
            "Make sure class_names.json belongs to this model."
        )

except Exception as e:

    print(
        "Could not check model output classes:",
        e
    )


# ============================================================
# PLANT NAME FUNCTION
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
# PLANT NAMES
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
print("Plants used:")

for plant in plant_names:

    print(" -", plant)


# ============================================================
# DISEASE INFORMATION
# ============================================================

disease_info = {

    # ========================================================
    # APPLE
    # ========================================================

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


    # ========================================================
    # BLUEBERRY
    # ========================================================

    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "care": (
            "The blueberry plant appears healthy. "
            "Maintain suitable soil moisture, "
            "sunlight and nutrition."
        )
    },


    # ========================================================
    # CHERRY
    # ========================================================

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


    # ========================================================
    # CORN
    # ========================================================

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


    # ========================================================
    # GRAPE
    # ========================================================

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


    # ========================================================
    # ORANGE
    # ========================================================

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


    # ========================================================
    # PEACH
    # ========================================================

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


    # ========================================================
    # PEPPER
    # ========================================================

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


    # ========================================================
    # POTATO
    # ========================================================

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


    # ========================================================
    # RASPBERRY
    # ========================================================

    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "care": (
            "The raspberry plant appears healthy. "
            "Maintain good sunlight, airflow "
            "and soil moisture."
        )
    },


    # ========================================================
    # SOYBEAN
    # ========================================================

    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "care": (
            "The soybean plant appears healthy. "
            "Continue proper irrigation, nutrition "
            "and regular crop monitoring."
        )
    },


    # ========================================================
    # SQUASH
    # ========================================================

    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "care": (
            "Remove severely affected leaves and "
            "improve airflow. Avoid excessive humidity "
            "and prolonged leaf wetness."
        )
    },


    # ========================================================
    # STRAWBERRY
    # ========================================================

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


    # ========================================================
    # TOMATO
    # ========================================================

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

    # Green pixels
    green_pixels = (
        (g > r * 1.05) &
        (g > b * 1.03) &
        (g > 45)
    )

    green_ratio = np.mean(
        green_pixels
    )

    # Brown / dry leaf pixels
    brown_pixels = (
        (r > g * 1.05) &
        (g > b * 1.10) &
        (r > 50) &
        (g > 35)
    )

    brown_ratio = np.mean(
        brown_pixels
    )

    brightness = np.mean(
        (r + g + b) / 3
    )

    print(
        "Green pixel ratio:",
        round(float(green_ratio * 100), 2),
        "%"
    )

    print(
        "Brown pixel ratio:",
        round(float(brown_ratio * 100), 2),
        "%"
    )

    print(
        "Average brightness:",
        round(float(brightness), 2)
    )

    if green_ratio >= 0.06:
        return True

    if brown_ratio >= 0.08:
        return True

    return False


# ============================================================
# COMMON ERROR PAGE
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
            "Please select a plant image."
        )

    file = request.files["image"]

    if file.filename == "":

        return show_error(
            "Please select a plant image."
        )

    try:

        # ====================================================
        # OPEN IMAGE
        # ====================================================

        original_image = Image.open(
            file.stream
        ).convert("RGB")

        print(
            "Original image size:",
            original_image.size
        )


        # ====================================================
        # IMAGE SIZE CHECK
        # ====================================================

        if (
            original_image.width < 80
            or
            original_image.height < 80
        ):

            print(
                "IMAGE REJECTED - IMAGE TOO SMALL"
            )

            return show_error(
                "Image could not be reliably identified. "
                "Please upload a clear leaf image."
            )


        # ====================================================
        # PLANT IMAGE CHECK
        # ====================================================

        plant_like = looks_like_plant_image(
            original_image
        )

        if not plant_like:

            print(
                "IMAGE REJECTED - DOES NOT LOOK LIKE A PLANT"
            )

            return show_error(
                "Image could not be reliably identified. "
                "Please upload a clear leaf image."
            )


        # ====================================================
        # PREPARE IMAGE
        # ====================================================

        image_array = prepare_image(
            original_image
        )

        print(
            "Model input shape:",
            image_array.shape
        )


        # ====================================================
        # MODEL PREDICTION
        # Render memory optimization:
        # Use direct inference instead of model.predict()
        # ====================================================

        predictions = model(
            image_array,
            training=False
        )

        probabilities = predictions.numpy()[0]

        # Free temporary TensorFlow prediction object
        del predictions


        # ====================================================
        # SAFETY CHECK
        # ====================================================

        if len(probabilities) != len(class_names):

            del image_array
            gc.collect()

            print(
                "ERROR: Model output and class names "
                "do not match."
            )

            return show_error(
                "The AI model configuration is incorrect. "
                "Please check the model and class_names.json."
            )


        # ====================================================
        # TOP 5 PREDICTIONS
        # ====================================================

        top_indices = np.argsort(
            probabilities
        )[-5:][::-1]

        print()
        print("TOP 5 AI PREDICTIONS")
        print("-" * 60)

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            print(
                rank,
                ".",
                class_names[int(index)],
                "->",
                round(
                    float(
                        probabilities[index] * 100
                    ),
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
        # TOP-2 MARGIN
        # ====================================================

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
            "FINAL PREDICTION:",
            predicted_class
        )

        print(
            "CONFIDENCE:",
            round(confidence, 2),
            "%"
        )

        print(
            "TOP-2 MARGIN:",
            round(margin, 2),
            "%"
        )


        # ====================================================
        # CONFIDENCE CHECK
        # ====================================================

        if confidence < CONFIDENCE_THRESHOLD:

            del image_array
            del probabilities
            gc.collect()

            print(
                "IMAGE REJECTED - LOW CONFIDENCE"
            )

            return show_error(
                "Image could not be reliably identified. "
                "Please upload a clear leaf image."
            )


        # ====================================================
        # MARGIN CHECK
        # ====================================================

        if margin < MARGIN_THRESHOLD:

            del image_array
            del probabilities
            gc.collect()

            print(
                "IMAGE REJECTED - UNCERTAIN PREDICTION"
            )

            return show_error(
                "The AI is uncertain about this image. "
                "Please upload a clear leaf image."
            )


        # ====================================================
        # DISEASE INFORMATION
        # ====================================================

        info = disease_info.get(
            predicted_class
        )

        if info is None:

            plant = get_plant_name(
                predicted_class
            )

            disease = (
                predicted_class
                .split("___")[-1]
                .replace("_", " ")
            )

            care = (
                "Monitor the plant regularly "
                "and consult a local agricultural "
                "expert if symptoms continue."
            )

        else:

            plant = info["plant"]

            disease = info["disease"]

            care = info["care"]


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
        # SUCCESS LOG
        # ====================================================

        print()
        print("IMAGE ACCEPTED")

        print("Plant:", plant)

        print("Disease:", disease)

        print("Severity:", severity)

        print("=" * 60)


        # ====================================================
        # MEMORY CLEANUP
        # ====================================================

        del image_array
        del probabilities
        del top_indices
        del sorted_probabilities

        gc.collect()


        # ====================================================
        # SHOW RESULT
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


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print()
        print("=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)

        print(
            str(e)
        )

        print("=" * 60)

        gc.collect()

        return show_error(
            "Unable to analyze this image. "
            "Please try another clear plant image."
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {

        "status": "online",

        "model": "Plant Disease MobileNetV2",

        "classes": len(
            class_names
        ),

        "plants": len(
            plant_names
        ),

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "margin_threshold":
            MARGIN_THRESHOLD
    }


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

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
        "Confidence threshold:",
        CONFIDENCE_THRESHOLD,
        "%"
    )

    print(
        "Top-2 margin threshold:",
        MARGIN_THRESHOLD,
        "%"
    )

    print()
    print("Server starting...")
    print()
    print("Computer:")
    print("http://127.0.0.1:5000")

    print()
    print("Phone:")
    print("http://192.168.43.248:5000")

    print("=" * 60)

    # IMPORTANT FOR SPYDER
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )
