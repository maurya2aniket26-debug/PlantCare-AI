from flask import Flask, render_template, request, session, redirect, url_for
import os
import json

# ============================================================
# TENSORFLOW RESOURCE SETTINGS
# MUST BE BEFORE IMPORTING TENSORFLOW
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
# SECRET KEY
# ============================================================

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "plantai_secret_key_2026"
)


# ============================================================
# LOGIN SETTINGS
# ============================================================

LOGIN_USERNAME = os.environ.get(
    "LOGIN_USERNAME",
    "admin"
)

LOGIN_PASSWORD = os.environ.get(
    "LOGIN_PASSWORD",
    "1234"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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
# AI SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 65.0
MARGIN_THRESHOLD = 10.0

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("AI PLANT CARE SYSTEM")
print("=" * 60)

print("Checking model...")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        "MODEL FILE NOT FOUND: " + MODEL_PATH
    )

if not os.path.isfile(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        "CLASS NAMES FILE NOT FOUND: " +
        CLASS_NAMES_PATH
    )


print("Model path:")
print(MODEL_PATH)

print("Class names path:")
print(CLASS_NAMES_PATH)

print("Loading TensorFlow model...")

try:

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully.")

except Exception as e:

    print("=" * 60)
    print("MODEL LOADING ERROR")
    print("=" * 60)
    print(str(e))
    print("=" * 60)

    raise


# ============================================================
# LOAD CLASS NAMES
# ============================================================

try:

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        class_names = json.load(f)

except Exception as e:

    print("=" * 60)
    print("CLASS NAMES ERROR")
    print("=" * 60)
    print(str(e))
    print("=" * 60)

    raise


# ============================================================
# NORMALIZE CLASS NAMES
# ============================================================

if isinstance(class_names, dict):

    # Support JSON such as:
    # {"0": "Apple___Apple_scab", ...}

    try:

        class_names = [
            class_names[str(i)]
            for i in range(len(class_names))
        ]

    except Exception:

        class_names = list(
            class_names.values()
        )


if not isinstance(class_names, list):

    raise ValueError(
        "class_names.json must contain a list "
        "of class names."
    )


class_names = [
    str(name)
    for name in class_names
]


print(
    "Number of classes:",
    len(class_names)
)


# ============================================================
# CHECK MODEL OUTPUT
# ============================================================

try:

    model_output_classes = int(
        model.output_shape[-1]
    )

    print(
        "Model output classes:",
        model_output_classes
    )

    print(
        "JSON classes:",
        len(class_names)
    )

    if model_output_classes != len(class_names):

        raise ValueError(
            "MODEL/CLASS COUNT MISMATCH: "
            f"Model has {model_output_classes} outputs "
            f"but class_names.json has "
            f"{len(class_names)} classes."
        )

except Exception as e:

    print("=" * 60)
    print("MODEL CONFIGURATION ERROR")
    print("=" * 60)
    print(str(e))
    print("=" * 60)

    raise


# ============================================================
# MODEL INPUT INFORMATION
# ============================================================

try:

    print(
        "Model input shape:",
        model.input_shape
    )

except Exception:

    pass


# ============================================================
# PLANT NAME FUNCTION
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
    list(
        set(
            get_plant_name(name)
            for name in class_names
        )
    )
)


print()
print("Plants detected:")

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
# PLANT IMAGE CHECK
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
    # GREEN PIXELS
    # --------------------------------------------------------

    green_pixels = (

        (g > r * 1.05)

        &

        (g > b * 1.03)

        &

        (g > 45)

    )


    green_ratio = np.mean(
        green_pixels
    )


    # --------------------------------------------------------
    # BROWN PIXELS
    # --------------------------------------------------------

    brown_pixels = (

        (r > g * 1.05)

        &

        (g > b * 1.10)

        &

        (r > 50)

        &

        (g > 35)

    )


    brown_ratio = np.mean(
        brown_pixels
    )


    brightness = np.mean(
        (r + g + b) / 3
    )


    print(
        "Green ratio:",
        round(
            float(green_ratio * 100),
            2
        ),
        "%"
    )

    print(
        "Brown ratio:",
        round(
            float(brown_ratio * 100),
            2
        ),
        "%"
    )

    print(
        "Brightness:",
        round(
            float(brightness),
            2
        )
    )


    if green_ratio >= 0.06:

        return True


    if brown_ratio >= 0.08:

        return True


    return False


# ============================================================
# ERROR PAGE
# ============================================================

def show_error(message):

    return render_template(
        "index.html",
        error=message,
        plant_names=plant_names,
        username=session.get("username")
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if session.get("logged_in"):

        return redirect(
            url_for("home")
        )


    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )


        if (
            username == LOGIN_USERNAME
            and
            password == LOGIN_PASSWORD
        ):

            session.clear()

            session["logged_in"] = True
            session["username"] = username

            print(
                "LOGIN SUCCESSFUL:",
                username
            )

            return redirect(
                url_for("home")
            )


        print(
            "LOGIN FAILED"
        )

        return render_template(
            "login.html",
            error="Invalid username or password."
        )


    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    username = session.get(
        "username",
        "Unknown"
    )

    print(
        "USER LOGGED OUT:",
        username
    )

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )


    return render_template(
        "index.html",
        plant_names=plant_names,
        username=session.get("username")
    )


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )


    print()
    print("=" * 60)
    print("NEW IMAGE RECEIVED")
    print("=" * 60)


    # ========================================================
    # CHECK IMAGE
    # ========================================================

    if "image" not in request.files:

        return show_error(
            "Please select a plant image."
        )


    file = request.files["image"]


    if not file or file.filename == "":

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
            "Original image:",
            original_image.size
        )


        # ====================================================
        # SIZE CHECK
        # ====================================================

        if (
            original_image.width < 80
            or
            original_image.height < 80
        ):

            return show_error(
                "Image is too small. "
                "Please upload a clear leaf image."
            )


        # ====================================================
        # PLANT CHECK
        # ====================================================

        if not looks_like_plant_image(
            original_image
        ):

            print(
                "IMAGE REJECTED: "
                "Does not look like a plant."
            )

            return show_error(
                "This image does not look like "
                "a plant leaf. Please upload a "
                "clear plant image."
            )


        # ====================================================
        # PREPARE IMAGE
        # ====================================================

        image_array = prepare_image(
            original_image
        )


        print(
            "Prepared image:",
            image_array.shape
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        predictions = model.predict(
            image_array,
            verbose=0
        )


        probabilities = np.asarray(
            predictions[0],
            dtype=np.float32
        )


        # ====================================================
        # SAFETY CHECK
        # ====================================================

        if len(probabilities) != len(class_names):

            print(
                "MODEL/CLASS NAME MISMATCH"
            )

            return show_error(
                "AI model configuration error. "
                "Please check plant_disease_model.keras "
                "and class_names.json."
            )


        # ====================================================
        # NORMALIZE PREDICTIONS IF NECESSARY
        # ====================================================

        probabilities = np.nan_to_num(
            probabilities,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


        # If model outputs logits rather than probabilities,
        # convert them to probabilities.

        total = float(
            np.sum(probabilities)
        )

        if (
            np.any(probabilities < 0)
            or
            abs(total - 1.0) > 0.01
        ):

            exp_values = np.exp(
                probabilities -
                np.max(probabilities)
            )

            probabilities = (
                exp_values /
                np.sum(exp_values)
            )


        # ====================================================
        # TOP 5
        # ====================================================

        top_indices = np.argsort(
            probabilities
        )[-5:][::-1]


        print()
        print(
            "TOP 5 PREDICTIONS"
        )

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
            np.argmax(
                probabilities
            )
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
        # TOP 2 MARGIN
        # ====================================================

        sorted_probabilities = np.sort(
            probabilities
        )


        top_probability = float(
            sorted_probabilities[-1]
        )


        if len(sorted_probabilities) > 1:

            second_probability = float(
                sorted_probabilities[-2]
            )

        else:

            second_probability = 0.0


        margin = (
            top_probability -
            second_probability
        ) * 100


        print()
        print(
            "FINAL:",
            predicted_class
        )

        print(
            "CONFIDENCE:",
            round(
                confidence,
                2
            ),
            "%"
        )

        print(
            "MARGIN:",
            round(
                margin,
                2
            ),
            "%"
        )


        # ====================================================
        # CONFIDENCE CHECK
        # ====================================================

        if confidence < CONFIDENCE_THRESHOLD:

            print(
                "REJECTED: LOW CONFIDENCE"
            )

            return show_error(
                "The AI could not identify this image "
                "with enough confidence. Please upload "
                "a clear close-up image of a plant leaf."
            )


        # ====================================================
        # MARGIN CHECK
        # ====================================================

        if margin < MARGIN_THRESHOLD:

            print(
                "REJECTED: UNCERTAIN"
            )

            return show_error(
                "The AI is uncertain about this image. "
                "Please upload a clearer plant leaf image."
            )


        # ====================================================
        # DISEASE INFORMATION
        # ====================================================

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
            "Confidence:",
            round(
                confidence,
                2
            ),
            "%"
        )

        print(
            "Severity:",
            severity
        )

        print("=" * 60)


        # ====================================================
        # RESULT PAGE
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

            plant_names=plant_names,

            username=session.get("username")

        )


    except Exception as e:

        print()
        print("=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)

        print(
            type(e).__name__
        )

        print(
            str(e)
        )

        print("=" * 60)


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

        "model": "plant_disease_model.keras",

        "classes": len(class_names),

        "plants": len(plant_names),

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "margin_threshold":
            MARGIN_THRESHOLD

    }


# ============================================================
# FAVICON
# Prevent unnecessary 404 messages
# ============================================================

@app.route("/favicon.ico")
def favicon():

    return "", 204


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return show_error(
        "The uploaded image is too large. "
        "Please choose a smaller image."
    )


@app.errorhandler(404)
def page_not_found(error):

    return redirect(
        url_for("home")
    )


@app.errorhandler(500)
def internal_server_error(error):

    print(
        "INTERNAL SERVER ERROR:",
        error
    )

    return render_template(
        "index.html",
        error=(
            "Something went wrong while processing "
            "the request. Please try again."
        ),
        plant_names=plant_names,
        username=session.get("username")
    ), 500


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
        "Confidence:",
        CONFIDENCE_THRESHOLD,
        "%"
    )

    print(
        "Margin:",
        MARGIN_THRESHOLD,
        "%"
    )

    print()
    print(
        "Username:",
        LOGIN_USERNAME
    )

    print(
        "Server starting..."
    )

    print("=" * 60)


    # ========================================================
    # RENDER + PHONE + LOCAL NETWORK
    # ========================================================

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

        use_reloader=False

    )
