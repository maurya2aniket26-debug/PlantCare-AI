import os
import json
import gc
import logging

# ============================================================
# RENDER / TENSORFLOW SETTINGS
# ============================================================

# Limit TensorFlow CPU usage on Render
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plant_disease_model.keras"
)

CLASSES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "classes.json"
)


# ============================================================
# TENSORFLOW CPU CONFIGURATION
# ============================================================

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


# ============================================================
# PLANT DISEASE CLASSES
# ============================================================

DEFAULT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",

    "Blueberry___healthy",

    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",

    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",

    "Orange___Haunglongbing_(Citrus_greening)",

    "Peach___Bacterial_spot",
    "Peach___healthy",

    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",

    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",

    "Raspberry___healthy",

    "Soybean___healthy",

    "Squash___Powdery_mildew",

    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",

    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_classes():

    if os.path.exists(CLASSES_PATH):

        try:

            with open(
                CLASSES_PATH,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            # Support several common JSON formats
            if isinstance(data, list):
                classes = data

            elif isinstance(data, dict):

                if "classes" in data:
                    classes = data["classes"]

                elif "class_names" in data:
                    classes = data["class_names"]

                else:
                    # If JSON is {"0":"Apple...", "1":"..."}
                    try:
                        classes = [
                            value
                            for key, value in sorted(
                                data.items(),
                                key=lambda x: int(x[0])
                            )
                        ]
                    except Exception:
                        classes = list(data.values())

            else:
                classes = DEFAULT_CLASSES

            if len(classes) > 0:
                return classes

        except Exception as e:

            print(
                "Warning: Could not read classes.json:",
                e
            )

    print(
        "Using built-in PlantVillage class list."
    )

    return DEFAULT_CLASSES


CLASS_NAMES = load_classes()


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("Loading Plant Disease AI model...")
print("=" * 60)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "Model file not found: " + MODEL_PATH
    )


try:

    MODEL = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully.")

except Exception as e:

    print("ERROR loading model:")
    print(e)

    raise


# ============================================================
# MODEL INFORMATION
# ============================================================

try:
    print(
        "Model input shape:",
        MODEL.input_shape
    )

except Exception:
    pass


MODEL_OUTPUT_CLASSES = None

try:

    output_shape = MODEL.output_shape

    if isinstance(output_shape, list):
        output_shape = output_shape[0]

    MODEL_OUTPUT_CLASSES = int(
        output_shape[-1]
    )

except Exception:

    MODEL_OUTPUT_CLASSES = len(CLASS_NAMES)


print(
    "Number of classes:",
    len(CLASS_NAMES)
)

print(
    "Model output classes:",
    MODEL_OUTPUT_CLASSES
)

print(
    "JSON classes:",
    len(CLASS_NAMES)
)


# ============================================================
# SAFETY CHECK
# ============================================================

if MODEL_OUTPUT_CLASSES != len(CLASS_NAMES):

    print("=" * 60)
    print("WARNING")
    print(
        "Model output classes and class names do not match."
    )
    print(
        "Model:",
        MODEL_OUTPUT_CLASSES
    )
    print(
        "Classes:",
        len(CLASS_NAMES)
    )
    print("=" * 60)

    # Use only matching number
    if MODEL_OUTPUT_CLASSES < len(CLASS_NAMES):

        CLASS_NAMES = CLASS_NAMES[
            :MODEL_OUTPUT_CLASSES
        ]


# ============================================================
# PLANTS
# ============================================================

def get_plant_name(class_name):

    if "___" not in class_name:
        return class_name

    plant = class_name.split(
        "___",
        1
    )[0]

    plant = plant.replace(
        "(including sour)",
        ""
    )

    plant = plant.replace(
        "(maize)",
        ""
    )

    plant = plant.replace(
        ",_bell",
        ""
    )

    plant = plant.replace(
        "_bell",
        ""
    )

    plant = plant.replace(
        ",",
        ""
    )

    plant = plant.strip()

    if plant == "Corn":
        return "Corn"

    if plant == "Corn_":
        return "Corn"

    if plant == "Pepper":
        return "Pepper"

    return plant


PLANT_NAMES = sorted(
    list(
        set(
            get_plant_name(name)
            for name in CLASS_NAMES
        )
    )
)


print()
print("Plants used:")

for plant in PLANT_NAMES:
    print(" -", plant)


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    # ---------------- APPLE ----------------

    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "care": (
            "Remove infected leaves and fallen plant debris. "
            "Improve air circulation by pruning crowded branches. "
            "Avoid keeping foliage wet for long periods. "
            "Use an appropriate fungicide according to its label "
            "when disease pressure is high."
        ),
        "status": "HIGH RISK"
    },

    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "care": (
            "Remove affected leaves, fruit and dead branches. "
            "Keep the area around the tree clean and dry. "
            "Prune damaged wood and improve air circulation. "
            "Use suitable disease-control treatment when necessary."
        ),
        "status": "HIGH RISK"
    },

    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "care": (
            "Remove heavily infected leaves and improve airflow. "
            "Avoid excessive leaf wetness. "
            "Monitor the plant during humid weather and use "
            "an appropriate fungicide if recommended."
        ),
        "status": "MODERATE RISK"
    },

    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "care": (
            "The plant appears healthy. Continue balanced watering, "
            "good sunlight, proper nutrition and regular inspection "
            "for early signs of disease."
        ),
        "status": "HEALTHY"
    },


    # ---------------- BLUEBERRY ----------------

    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "care": (
            "Maintain acidic, well-drained soil and regular moisture. "
            "Provide good sunlight and airflow. Remove damaged leaves "
            "and monitor the plant regularly."
        ),
        "status": "HEALTHY"
    },


    # ---------------- CHERRY ----------------

    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "care": (
            "Remove severely affected leaves. Improve sunlight and "
            "air circulation around the tree. Avoid excessive nitrogen "
            "fertilization and keep foliage from staying wet."
        ),
        "status": "MODERATE RISK"
    },

    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "care": (
            "Maintain good sunlight, balanced watering and airflow. "
            "Prune overcrowded growth and inspect leaves regularly."
        ),
        "status": "HEALTHY"
    },


    # ---------------- CORN ----------------

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn",
        "disease": "Gray Leaf Spot",
        "care": (
            "Remove heavily infected plant material where practical. "
            "Improve field airflow and avoid excessive moisture. "
            "Use resistant varieties and appropriate fungicide "
            "management when recommended."
        ),
        "status": "HIGH RISK"
    },

    "Corn_(maize)___Common_rust_": {
        "plant": "Corn",
        "disease": "Common Rust",
        "care": (
            "Monitor rust spots regularly. Maintain healthy plant "
            "nutrition and airflow. Avoid prolonged leaf wetness and "
            "consider appropriate fungicide management for severe cases."
        ),
        "status": "MODERATE RISK"
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "care": (
            "Remove infected debris after harvest and improve airflow. "
            "Use resistant varieties where possible. Monitor the crop "
            "closely and use suitable fungicide treatment when required."
        ),
        "status": "HIGH RISK"
    },

    "Corn_(maize)___healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "care": (
            "Maintain consistent soil moisture, sufficient sunlight "
            "and balanced nutrition. Monitor leaves regularly for "
            "rust or blight symptoms."
        ),
        "status": "HEALTHY"
    },


    # ---------------- GRAPE ----------------

    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "care": (
            "Remove infected leaves and fruit clusters. Improve canopy "
            "airflow through pruning and avoid prolonged leaf wetness. "
            "Use an appropriate fungicide when necessary."
        ),
        "status": "HIGH RISK"
    },

    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
        "care": (
            "Remove severely affected plant material where practical. "
            "Maintain good vineyard sanitation and avoid injuries "
            "during pruning. Monitor affected vines closely."
        ),
        "status": "HIGH RISK"
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight",
        "care": (
            "Remove severely affected leaves, improve ventilation and "
            "avoid excessive canopy humidity. Keep the vineyard clean "
            "and consider suitable fungicide management."
        ),
        "status": "MODERATE RISK"
    },

    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "care": (
            "Maintain good sunlight and airflow through proper canopy "
            "management. Water the root zone rather than keeping leaves "
            "wet and inspect vines regularly."
        ),
        "status": "HEALTHY"
    },


    # ---------------- ORANGE ----------------

    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Citrus Greening",
        "care": (
            "Remove severely affected plant material where recommended "
            "and monitor for insect vectors. Maintain good tree nutrition "
            "and consult local agricultural guidance for confirmed cases."
        ),
        "status": "HIGH RISK"
    },


    # ---------------- PEACH ----------------

    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "care": (
            "Remove badly affected leaves and fruit. Improve airflow and "
            "avoid overhead irrigation. Maintain balanced nutrition and "
            "follow local recommendations for bacterial disease control."
        ),
        "status": "HIGH RISK"
    },

    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "care": (
            "Maintain regular watering, sunlight and airflow. Prune "
            "overcrowded branches and inspect new growth regularly."
        ),
        "status": "HEALTHY"
    },


    # ---------------- PEPPER ----------------

    "Pepper,_bell___Bacterial_spot": {
        "plant": "Pepper",
        "disease": "Bacterial Spot",
        "care": (
            "Remove severely affected leaves and fruit. Avoid overhead "
            "watering and working with wet plants. Maintain good spacing "
            "and sanitation."
        ),
        "status": "HIGH RISK"
    },

    "Pepper,_bell___healthy": {
        "plant": "Pepper",
        "disease": "Healthy",
        "care": (
            "Maintain consistent soil moisture, good sunlight and airflow. "
            "Avoid prolonged leaf wetness and inspect the plant regularly."
        ),
        "status": "HEALTHY"
    },


    # ---------------- POTATO ----------------

    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "care": (
            "Remove heavily affected leaves and plant debris. Maintain "
            "balanced fertilization and adequate spacing. Avoid wetting "
            "foliage unnecessarily and consider suitable fungicide treatment."
        ),
        "status": "HIGH RISK"
    },

    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "care": (
            "Remove severely infected foliage and isolate affected plants "
            "when possible. Avoid prolonged leaf wetness and monitor nearby "
            "plants closely. Seek local agricultural guidance for fungicide use."
        ),
        "status": "CRITICAL"
    },

    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "care": (
            "Maintain even soil moisture, good drainage and sufficient "
            "sunlight. Monitor leaves regularly for early blight or "
            "late blight symptoms."
        ),
        "status": "HEALTHY"
    },


    # ---------------- RASPBERRY ----------------

    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "care": (
            "Provide well-drained soil, adequate sunlight and regular "
            "watering. Remove old or damaged canes and maintain airflow."
        ),
        "status": "HEALTHY"
    },


    # ---------------- SOYBEAN ----------------

    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "care": (
            "Maintain appropriate soil moisture and balanced nutrition. "
            "Provide good field spacing and monitor leaves regularly "
            "for disease symptoms."
        ),
        "status": "HEALTHY"
    },


    # ---------------- SQUASH ----------------

    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "care": (
            "Remove heavily affected leaves and improve air circulation. "
            "Keep foliage as dry as practical and avoid excessive nitrogen. "
            "Use an appropriate treatment when disease pressure increases."
        ),
        "status": "MODERATE RISK"
    },


    # ---------------- STRAWBERRY ----------------

    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "care": (
            "Remove severely damaged leaves and maintain consistent soil "
            "moisture. Avoid drought stress and improve airflow around plants."
        ),
        "status": "MODERATE RISK"
    },

    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "care": (
            "Maintain regular watering, good sunlight and well-drained soil. "
            "Remove old leaves and keep the growing area clean."
        ),
        "status": "HEALTHY"
    },


    # ---------------- TOMATO ----------------

    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "care": (
            "Remove severely affected leaves and fruit. Avoid overhead "
            "watering and improve plant spacing. Keep tools and growing "
            "areas clean and monitor nearby plants."
        ),
        "status": "HIGH RISK"
    },

    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "care": (
            "Remove infected lower leaves and fallen debris. Improve airflow "
            "by spacing plants properly. Water at the soil level and avoid "
            "keeping leaves wet for long periods."
        ),
        "status": "HIGH RISK"
    },

    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "care": (
            "Remove severely infected foliage and fruit. Improve ventilation "
            "and avoid prolonged leaf wetness. Monitor surrounding plants "
            "carefully and follow local guidance for disease control."
        ),
        "status": "CRITICAL"
    },

    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "care": (
            "Improve ventilation and reduce humidity around the foliage. "
            "Avoid overhead watering and remove heavily infected leaves. "
            "Keep sufficient space between plants."
        ),
        "status": "MODERATE RISK"
    },

    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "care": (
            "Remove affected lower leaves and plant debris. Water at the "
            "base and improve airflow. Avoid working with wet foliage."
        ),
        "status": "HIGH RISK"
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider Mites",
        "care": (
            "Inspect the underside of leaves carefully. Remove heavily "
            "affected leaves and reduce plant stress with appropriate "
            "watering. Use an appropriate mite-control treatment when needed."
        ),
        "status": "HIGH RISK"
    },

    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "care": (
            "Remove severely affected leaves and improve airflow. Avoid "
            "prolonged leaf wetness and keep the growing area clean."
        ),
        "status": "HIGH RISK"
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "care": (
            "Inspect plants for whitefly activity and remove severely "
            "affected plants where appropriate. Control insect vectors "
            "using locally recommended methods and keep the area clean."
        ),
        "status": "CRITICAL"
    },

    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "care": (
            "Remove severely infected plants when appropriate and disinfect "
            "hands and tools after handling affected plants. Avoid spreading "
            "plant sap between healthy and infected plants."
        ),
        "status": "CRITICAL"
    },

    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "care": (
            "The tomato plant appears healthy. Maintain consistent watering, "
            "good sunlight, balanced nutrition and good airflow. Continue "
            "regular inspection for early symptoms."
        ),
        "status": "HEALTHY"
    }
}


# ============================================================
# GENERIC INFORMATION FALLBACK
# ============================================================

def get_disease_info(class_name):

    if class_name in DISEASE_INFO:

        return DISEASE_INFO[class_name]

    plant = get_plant_name(class_name)

    if "___" in class_name:

        disease_name = class_name.split(
            "___",
            1
        )[1]

        disease_name = disease_name.replace(
            "_",
            " "
        )

        disease_name = disease_name.replace(
            "(",
            ""
        )

        disease_name = disease_name.replace(
            ")",
            ""
        )

    else:

        disease_name = class_name

    if disease_name.lower().strip() == "healthy":

        return {
            "plant": plant,
            "disease": "Healthy",
            "care": (
                "The plant appears healthy. Maintain proper watering, "
                "sunlight, nutrition and airflow, and continue regular "
                "monitoring."
            ),
            "status": "HEALTHY"
        }

    return {
        "plant": plant,
        "disease": disease_name.title(),
        "care": (
            "Remove severely affected plant material where practical. "
            "Maintain good airflow, suitable watering and clean growing "
            "conditions. Monitor the plant closely and consult local "
            "agricultural guidance if symptoms continue."
        ),
        "status": "DISEASE DETECTED"
    }


# ============================================================
# IMAGE VALIDATION
# ============================================================

def analyze_image_visuals(image):

    try:

        # Small image only for visual checks
        check_image = image.copy()

        check_image.thumbnail(
            (256, 256)
        )

        arr = np.asarray(
            check_image.convert("RGB"),
            dtype=np.float32
        )

        if arr.size == 0:
            return False

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        brightness = np.mean(
            (r + g + b) / 3.0
        )

        # Basic green vegetation detection
        green_mask = (
            (g > r * 1.05) &
            (g > b * 1.02) &
            (g > 45)
        )

        green_ratio = (
            np.mean(green_mask) * 100
        )

        # Brown / yellow leaf areas
        brown_mask = (
            (r > b * 1.25) &
            (g > b * 1.05) &
            (r > 60)
        )

        brown_ratio = (
            np.mean(brown_mask) * 100
        )

        print(
            f"Green pixel ratio: {green_ratio:.1f} %"
        )

        print(
            f"Brown pixel ratio: {brown_ratio:.2f} %"
        )

        print(
            f"Average brightness: {brightness:.2f}"
        )

        # IMPORTANT:
        # Do not reject simply because green ratio is low.
        #
        # PlantVillage contains leaves with many colors:
        # green, yellow, brown, pale, diseased, etc.
        #
        # Therefore validation is intentionally tolerant.

        if brightness < 8:

            return False

        if brightness > 252:

            # Very bright image can still be a plant,
            # so do not reject automatically.
            pass

        return True

    except Exception as e:

        print(
            "Visual image check error:",
            e
        )

        # Never break prediction because of the
        # optional visual check.
        return True


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image):

    # Force RGB
    image = image.convert("RGB")

    # Resize directly to MobileNetV2 input size
    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    # Batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image):

    input_array = prepare_image(
        image
    )

    print(
        "Model input shape:",
        input_array.shape
    )

    # Direct inference
    predictions = MODEL(
        input_array,
        training=False
    )

    predictions = predictions.numpy()

    # Some models return logits instead of probabilities.
    # Convert safely if necessary.
    if predictions.ndim == 2:

        predictions = predictions[0]

    else:

        predictions = np.asarray(
            predictions
        ).reshape(-1)

    # Check whether values look like probabilities
    probability_sum = float(
        np.sum(predictions)
    )

    if (
        np.min(predictions) < 0
        or np.max(predictions) > 1.0
        or abs(probability_sum - 1.0) > 0.05
    ):

        predictions = tf.nn.softmax(
            predictions
        ).numpy()

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )

    # Ensure class count matches
    count = min(
        len(predictions),
        len(CLASS_NAMES)
    )

    predictions = predictions[
        :count
    ]

    classes = CLASS_NAMES[
        :count
    ]

    # Top 5
    top_indices = np.argsort(
        predictions
    )[::-1][:5]

    top_predictions = []

    for index in top_indices:

        top_predictions.append({
            "class": classes[index],
            "confidence": float(
                predictions[index] * 100
            )
        })

    return top_predictions


# ============================================================
# SEVERITY
# ============================================================

def calculate_severity(
    class_name,
    confidence
):

    info = get_disease_info(
        class_name
    )

    disease = info["disease"]

    if disease.lower() == "healthy":

        return "Healthy"

    if "CRITICAL" in info["status"]:

        return "High"

    if confidence >= 85:

        return "High"

    if confidence >= 70:

        return "Moderate"

    return "Mild"


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return render_template(
        "index.html",
        plant=None,
        disease=None,
        confidence=None,
        severity=None,
        care=None,
        status=None,
        plant_names=PLANT_NAMES,
        error=None
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

    try:

        if "image" not in request.files:

            return render_template(
                "index.html",
                plant=None,
                disease=None,
                confidence=None,
                severity=None,
                care=None,
                status=None,
                plant_names=PLANT_NAMES,
                error="Please select a plant image."
            )

        file = request.files["image"]

        if file.filename == "":

            return render_template(
                "index.html",
                plant=None,
                disease=None,
                confidence=None,
                severity=None,
                care=None,
                status=None,
                plant_names=PLANT_NAMES,
                error="Please select an image."
            )

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        image = Image.open(
            file.stream
        )

        print(
            "Original image size:",
            image.size
        )

        # ----------------------------------------------------
        # VISUAL CHECK
        # ----------------------------------------------------

        looks_like_plant = analyze_image_visuals(
            image
        )

        if not looks_like_plant:

            print(
                "IMAGE REJECTED - INVALID IMAGE"
            )

            return render_template(
                "index.html",
                plant=None,
                disease=None,
                confidence=None,
                severity=None,
                care=None,
                status=None,
                plant_names=PLANT_NAMES,
                error=(
                    "The uploaded image could not be analyzed. "
                    "Please upload a clear plant or leaf image."
                )
            )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        top_predictions = predict_image(
            image
        )

        if not top_predictions:

            raise RuntimeError(
                "The AI model returned no prediction."
            )

        # ----------------------------------------------------
        # PRINT TOP 5
        # ----------------------------------------------------

        print()
        print(
            "TOP 5 AI PREDICTIONS"
        )

        print(
            "-" * 60
        )

        for i, result in enumerate(
            top_predictions,
            start=1
        ):

            print(
                f"{i} . "
                f"{result['class']} -> "
                f"{result['confidence']:.2f} %"
            )

        print(
            "-" * 60
        )

        # ----------------------------------------------------
        # FINAL PREDICTION
        # ----------------------------------------------------

        best = top_predictions[0]

        predicted_class = best["class"]

        confidence = best["confidence"]

        if len(top_predictions) >= 2:

            second_confidence = (
                top_predictions[1]["confidence"]
            )

        else:

            second_confidence = 0.0

        top2_margin = (
            confidence - second_confidence
        )

        print()
        print(
            "FINAL PREDICTION:",
            predicted_class
        )

        print(
            f"CONFIDENCE: {confidence:.2f} %"
        )

        print(
            f"TOP-2 MARGIN: {top2_margin:.2f} %"
        )

        # ----------------------------------------------------
        # CONFIDENCE CHECK
        # ----------------------------------------------------

        CONFIDENCE_THRESHOLD = 65.0

        MARGIN_THRESHOLD = 10.0

        if (
            confidence < CONFIDENCE_THRESHOLD
            or top2_margin < MARGIN_THRESHOLD
        ):

            print()
            print(
                "IMAGE REJECTED - LOW AI CONFIDENCE"
            )

            return render_template(
                "index.html",
                plant=None,
                disease=None,
                confidence=None,
                severity=None,
                care=None,
                status=None,
                plant_names=PLANT_NAMES,
                error=(
                    "The AI is not confident enough about this image. "
                    f"Confidence: {confidence:.1f}%. "
                    "Please upload a clearer close-up image of the leaf."
                )
            )

        # ----------------------------------------------------
        # DISEASE INFORMATION
        # ----------------------------------------------------

        info = get_disease_info(
            predicted_class
        )

        plant = info["plant"]

        disease = info["disease"]

        care = info["care"]

        status = info["status"]

        severity = calculate_severity(
            predicted_class,
            confidence
        )

        print()
        print(
            "IMAGE ACCEPTED"
        )

        print(
            "Plant:",
            plant
        )

        print(
            "Disease:",
            disease
        )

        print(
            "Severity:",
            severity
        )

        # ----------------------------------------------------
        # MEMORY CLEANUP
        # ----------------------------------------------------

        del image

        del top_predictions

        gc.collect()

        # ----------------------------------------------------
        # RENDER RESULT
        # ----------------------------------------------------

        return render_template(
            "index.html",
            plant=plant,
            disease=disease,
            confidence=round(
                confidence,
                2
            ),
            severity=severity,
            care=care,
            status=status,
            plant_names=PLANT_NAMES,
            error=None
        )

    except Exception as e:

        print()
        print(
            "=" * 60
        )

        print(
            "PREDICTION ERROR"
        )

        print(
            "=" * 60
        )

        print(
            type(e).__name__,
            ":",
            str(e)
        )

        print(
            "=" * 60
        )

        gc.collect()

        return render_template(
            "index.html",
            plant=None,
            disease=None,
            confidence=None,
            severity=None,
            care=None,
            status=None,
            plant_names=PLANT_NAMES,
            error=(
                "Unable to analyze this image. "
                "Please try another clear plant image."
            )
        )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(
    413
)
def file_too_large(error):

    return render_template(
        "index.html",
        plant=None,
        disease=None,
        confidence=None,
        severity=None,
        care=None,
        status=None,
        plant_names=PLANT_NAMES,
        error=(
            "Image is too large. "
            "Please upload an image smaller than 8 MB."
        )
    ), 413


@app.errorhandler(
    500
)
def internal_error(error):

    gc.collect()

    return render_template(
        "index.html",
        plant=None,
        disease=None,
        confidence=None,
        severity=None,
        care=None,
        status=None,
        plant_names=PLANT_NAMES,
        error=(
            "The server could not process the image. "
            "Please try again."
        )
    ), 500


# ============================================================
# RENDER STARTUP
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
    print(
        "AI PLANT CARE SYSTEM"
    )
    print("=" * 60)

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Classes:",
        len(CLASS_NAMES)
    )

    print(
        "Plants:",
        len(PLANT_NAMES)
    )

    print(
        "Confidence threshold:",
        "65.0 %"
    )

    print(
        "Top-2 margin threshold:",
        "10.0 %"
    )

    print()
    print(
        "Server starting..."
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
        threaded=False
    )
