from flask import (
    Flask,
    render_template,
    request,
    session,
    redirect,
    url_for
)

import os
import json
import sqlite3
import time

import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# ============================================================
# SECRET KEY
# ============================================================

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "plantai_secret_key_change_this_2026"
)

# ============================================================
# SERVER START ID
# A new ID is created every time Flask starts.
# This forces the user to login again after a restart.
# ============================================================

SERVER_START_ID = str(time.time())


# ============================================================
# LOGIN SESSION CHECK
# ============================================================

def is_logged_in():

    if not session.get("logged_in"):
        return False

    # If Flask has restarted, the old browser session
    # is no longer considered valid.
    if session.get("server_start_id") != SERVER_START_ID:

        session.clear()

        return False

    return True

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plant_disease_model.tflite"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.json"
)

# ============================================================
# DATABASE SETTINGS
# ============================================================

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "plantai_users.db"
)

DATABASE_URL = os.environ.get(
    "DATABASE_URL"
)

# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    # --------------------------------------------------------
    # POSTGRESQL
    # --------------------------------------------------------

    if DATABASE_URL:

        import psycopg2

        database_url = DATABASE_URL

        if database_url.startswith("postgres://"):

            database_url = database_url.replace(
                "postgres://",
                "postgresql://",
                1
            )

        connection = psycopg2.connect(
            database_url
        )

        return connection

    # --------------------------------------------------------
    # SQLITE
    # --------------------------------------------------------

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

def init_database():

    connection = get_db_connection()
    cursor = connection.cursor()

    # ========================================================
    # USERS TABLE
    # ========================================================

    if DATABASE_URL:

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (

                id SERIAL PRIMARY KEY,

                username VARCHAR(50)
                UNIQUE NOT NULL,

                password_hash TEXT NOT NULL,

                created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP

            )
            """
        )

    else:

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                username TEXT
                UNIQUE NOT NULL,

                password_hash TEXT NOT NULL,

                created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP

            )
            """
        )

    # ========================================================
    # HISTORY TABLE
    # ========================================================

    if DATABASE_URL:

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS history (

                id SERIAL PRIMARY KEY,

                user_id INTEGER NOT NULL,

                plant VARCHAR(100) NOT NULL,

                disease VARCHAR(200) NOT NULL,

                confidence REAL NOT NULL,

                severity VARCHAR(50),

                status VARCHAR(100),

                care TEXT,

                created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP

            )
            """
        )

    else:

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS history (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                user_id INTEGER NOT NULL,

                plant TEXT NOT NULL,

                disease TEXT NOT NULL,

                confidence REAL NOT NULL,

                severity TEXT,

                status TEXT,

                care TEXT,

                created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP

            )
            """
        )

    connection.commit()

    cursor.close()
    connection.close()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

print("=" * 60)
print("Initializing PlantAI database...")
print("=" * 60)

try:

    init_database()

    if DATABASE_URL:

        print(
            "Database: PostgreSQL"
        )

    else:

        print(
            "Database: SQLite"
        )

        print(
            "Database file:",
            DATABASE_PATH
        )

    print(
        "Users table ready."
    )

    print(
        "History table ready."
    )

except Exception as e:

    print(
        "DATABASE INITIALIZATION ERROR:"
    )

    print(
        str(e)
    )

    raise


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 65.0

MARGIN_THRESHOLD = 10.0

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL (TFLite)
# ============================================================

print("=" * 60)
print("Loading Plant Disease AI model (TFLite)...")
print("=" * 60)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "Model not found:\n" +
        MODEL_PATH
    )

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(
    "Model loaded successfully."
)

print(
    "Model input shape:",
    input_details[0]["shape"]
)


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

print(
    "Number of classes:",
    len(class_names)
)


# ============================================================
# CHECK MODEL / CLASS COUNT
# ============================================================

try:

    model_output_classes = (
        output_details[0]["shape"][-1]
    )

    print(
        "Model output classes:",
        model_output_classes
    )

    print(
        "JSON classes:",
        len(class_names)
    )

    if (
        model_output_classes
        !=
        len(class_names)
    ):

        print()
        print("WARNING!")

        print(
            "Model classes and JSON classes "
            "do not match."
        )

        print(
            "Make sure class_names.json "
            "belongs to this model."
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

        plant_name = (
            class_name.split("___")[0]
        )

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

    print(
        " -",
        plant
    )


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
        "disease": (
            "Cercospora Leaf Spot / Gray Leaf Spot"
        ),
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
        "disease": (
            "Huanglongbing / Citrus Greening"
        ),
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
        "disease": (
            "Tomato Yellow Leaf Curl Virus"
        ),
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
# RECOMMENDED ENVIRONMENTAL CONDITIONS
# These are recommendations, NOT live sensor readings.
# ============================================================

environment_info = {

    "Apple": {
        "sunlight": "6–8h",
        "soil_moisture": "Moderate",
        "humidity": "50–70%",
        "temperature": "18–27°C"
    },

    "Blueberry": {
        "sunlight": "6–8h",
        "soil_moisture": "Moderate",
        "humidity": "50–70%",
        "temperature": "16–24°C"
    },

    "Cherry": {
        "sunlight": "6–8h",
        "soil_moisture": "Moderate",
        "humidity": "50–70%",
        "temperature": "18–25°C"
    },

    "Corn": {
        "sunlight": "6–8h",
        "soil_moisture": "Moderate",
        "humidity": "50–70%",
        "temperature": "20–30°C"
    },

    "Grape": {
        "sunlight": "7–8h",
        "soil_moisture": "Moderate",
        "humidity": "50–70%",
        "temperature": "20–30°C"
    },

    "Orange": {
        "sunlight": "6–8h",
        "soil_moisture": "Moderate",
        "humidity": "50–70%",
        "temperature": "20–30°C"
    },

    "Peach": {
        "sunlight": "6–8h",
        "soil_moisture": "Moderate",
        "humidity": "50–70%",
        "temperature": "18–27°C"
    },

    "Bell Pepper": {
        "sunlight": "6–8h",
        "soil_moisture": "Moderate",
        "humidity": "50–70%",
        "temperature": "20–28°C"
    },

    "Potato": {
        "sunlight": "6–8h",
        "soil_moisture": "Moderate",
        "humidity": "50–70%",
        "temperature": "15–24°C"
    },

    "Raspberry": {
        "sunlight": "6–8h",
        "soil_moisture": "Moderate",
        "humidity": "50–70%",
        "temperature": "16–24°C"
    },

    "Soybean": {
        "sunlight": "6–8h",
        "soil_moisture": "Moderate",
        "humidity": "50–70%",
        "temperature": "20–30°C"
    },

    "Squash": {
        "sunlight": "6–8h",
        "soil_moisture": "Moderate",
        "humidity": "50–70%",
        "temperature": "18–30°C"
    },

    "Strawberry": {
        "sunlight": "6–8h",
        "soil_moisture": "Moderate",
        "humidity": "60–70%",
        "temperature": "15–25°C"
    },

    "Tomato": {
        "sunlight": "6–8h",
        "soil_moisture": "Moderate",
        "humidity": "60–70%",
        "temperature": "20–28°C"
    }
}


# ============================================================
# PLANT HEALTH SCORE
# Based on AI diagnosis and confidence.
# This is an AI-derived score, NOT a physical sensor reading.
# ============================================================

def calculate_health_score(
    disease,
    confidence,
    severity
):

    if disease.lower() == "healthy":

        score = 70 + (
            confidence * 0.30
        )

    elif severity == "Moderate":

        score = 45 + (
            (100 - confidence) * 0.20
        )

    elif severity == "High":

        score = 25 + (
            (100 - confidence) * 0.15
        )

    else:

        score = 40

    score = max(
        0,
        min(
            100,
            score
        )
    )

    score = round(
        score
    )

    # --------------------------------------------------------
    # HEALTH CATEGORY
    # --------------------------------------------------------

    if score >= 75:

        health_status = "HEALTHY"

    elif score >= 50:

        health_status = "NEEDS ATTENTION"

    else:

        health_status = "CRITICAL"

    # --------------------------------------------------------
    # BREAKDOWN
    # --------------------------------------------------------

    if health_status == "HEALTHY":

        healthy_percent = score

        attention_percent = 100 - score

        critical_percent = 0

    elif health_status == "NEEDS ATTENTION":

        healthy_percent = max(
            0,
            score - 20
        )

        attention_percent = 100 - healthy_percent

        critical_percent = 0

    else:

        critical_percent = max(
            10,
            100 - score
        )

        attention_percent = 25

        healthy_percent = max(
            0,
            100 -
            attention_percent -
            critical_percent
        )

    return {
        "score": score,
        "status": health_status,
        "healthy": healthy_percent,
        "attention": attention_percent,
        "critical": critical_percent
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
        "Green pixel ratio:",
        round(
            float(
                green_ratio * 100
            ),
            2
        ),
        "%"
    )

    print(
        "Brown pixel ratio:",
        round(
            float(
                brown_ratio * 100
            ),
            2
        ),
        "%"
    )

    print(
        "Average brightness:",
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
        username=session.get("username"),
        history_page=False,
        health_score=None,
        health_status=None,
        health_healthy=None,
        health_attention=None,
        health_critical=None,
        environment=None
    )


# ============================================================
# GET CURRENT USER HISTORY
# ============================================================

def get_user_history():

    if not is_logged_in():

        return []

    user_id = session.get(
        "user_id"
    )

    connection = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        if DATABASE_URL:

            cursor.execute(
                """
                SELECT
                    id,
                    plant,
                    disease,
                    confidence,
                    severity,
                    status,
                    care,
                    created_at
                FROM history
                WHERE user_id = %s
                ORDER BY created_at DESC, id DESC
                """,
                (user_id,)
            )

            rows = cursor.fetchall()

            history = []

            for row in rows:

                history.append({

                    "id": row[0],

                    "plant": row[1],

                    "disease": row[2],

                    "confidence": row[3],

                    "severity": row[4],

                    "status": row[5],

                    "care": row[6],

                    "created_at": row[7]

                })

        else:

            cursor.execute(
                """
                SELECT
                    id,
                    plant,
                    disease,
                    confidence,
                    severity,
                    status,
                    care,
                    created_at
                FROM history
                WHERE user_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (user_id,)
            )

            rows = cursor.fetchall()

            history = []

            for row in rows:

                history.append({

                    "id": row["id"],

                    "plant": row["plant"],

                    "disease": row["disease"],

                    "confidence": row["confidence"],

                    "severity": row["severity"],

                    "status": row["status"],

                    "care": row["care"],

                    "created_at": row["created_at"]

                })

        cursor.close()

        connection.close()

        return history

    except Exception as e:

        print()
        print(
            "HISTORY LOAD ERROR:"
        )

        print(
            str(e)
        )

        if connection:

            try:
                connection.close()

            except:
                pass

        return []


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if is_logged_in():

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

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if not username:

            return render_template(
                "login.html",
                mode="register",
                error="Please enter a username.",
                username=username
            )

        if len(username) < 3:

            return render_template(
                "login.html",
                mode="register",
                error=(
                    "Username must contain at least 3 characters."
                ),
                username=username
            )

        if len(username) > 50:

            return render_template(
                "login.html",
                mode="register",
                error=(
                    "Username must be 50 characters or less."
                ),
                username=username
            )

        if not all(
            char.isalnum() or char in "_-"
            for char in username
        ):

            return render_template(
                "login.html",
                mode="register",
                error=(
                    "Username can contain only letters, "
                    "numbers, underscore and hyphen."
                ),
                username=username
            )

        if len(password) < 6:

            return render_template(
                "login.html",
                mode="register",
                error=(
                    "Password must contain at least 6 characters."
                ),
                username=username
            )

        if password != confirm_password:

            return render_template(
                "login.html",
                mode="register",
                error="Passwords do not match.",
                username=username
            )

        password_hash = generate_password_hash(
            password
        )

        connection = None

        try:

            connection = get_db_connection()

            cursor = connection.cursor()

            if DATABASE_URL:

                cursor.execute(
                    """
                    SELECT id
                    FROM users
                    WHERE LOWER(username) = LOWER(%s)
                    """,
                    (username,)
                )

            else:

                cursor.execute(
                    """
                    SELECT id
                    FROM users
                    WHERE LOWER(username) = LOWER(?)
                    """,
                    (username,)
                )

            existing_user = cursor.fetchone()

            if existing_user:

                cursor.close()
                connection.close()

                return render_template(
                    "login.html",
                    mode="register",
                    error=(
                        "That username already exists. "
                        "Please choose another username."
                    ),
                    username=username
                )

            if DATABASE_URL:

                cursor.execute(
                    """
                    INSERT INTO users
                    (
                        username,
                        password_hash
                    )
                    VALUES (%s, %s)
                    """,
                    (
                        username,
                        password_hash
                    )
                )

            else:

                cursor.execute(
                    """
                    INSERT INTO users
                    (
                        username,
                        password_hash
                    )
                    VALUES (?, ?)
                    """,
                    (
                        username,
                        password_hash
                    )
                )

            connection.commit()

            cursor.close()
            connection.close()

            print()
            print(
                "=" * 60
            )

            print(
                "NEW USER REGISTERED"
            )

            print(
                "Username:",
                username
            )

            print(
                "=" * 60
            )

            return render_template(
                "login.html",
                mode="login",
                success=(
                    "Account created successfully. "
                    "You can now log in."
                )
            )

        except Exception as e:

            print()
            print(
                "REGISTRATION ERROR:"
            )

            print(
                str(e)
            )

            if connection:

                try:

                    connection.rollback()
                    connection.close()

                except:
                    pass

            return render_template(
                "login.html",
                mode="register",
                error=(
                    "Unable to create the account. "
                    "Please try again."
                ),
                username=username
            )

    return render_template(
        "login.html",
        mode="register"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if is_logged_in():

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

        if not username or not password:

            return render_template(
                "login.html",
                mode="login",
                error=(
                    "Please enter username and password."
                ),
                username=username
            )

        connection = None

        try:

            connection = get_db_connection()

            cursor = connection.cursor()

            if DATABASE_URL:

                cursor.execute(
                    """
                    SELECT
                        id,
                        username,
                        password_hash
                    FROM users
                    WHERE LOWER(username) = LOWER(%s)
                    """,
                    (username,)
                )

            else:

                cursor.execute(
                    """
                    SELECT
                        id,
                        username,
                        password_hash
                    FROM users
                    WHERE LOWER(username) = LOWER(?)
                    """,
                    (username,)
                )

            user = cursor.fetchone()

            cursor.close()
            connection.close()

            if user:

                if DATABASE_URL:

                    stored_username = user[1]

                    stored_password_hash = user[2]

                else:

                    stored_username = user["username"]

                    stored_password_hash = user["password_hash"]

                if check_password_hash(
                    stored_password_hash,
                    password
                ):

                    session.clear()

                    session["logged_in"] = True

                    session["user_id"] = user[0]

                    session["username"] = (
                        stored_username
                    )

                    # Mark this session as belonging to
                    # the current Flask server instance.
                    session["server_start_id"] = SERVER_START_ID

                    print()
                    print(
                        "=" * 60
                    )

                    print(
                        "USER LOGIN SUCCESSFUL"
                    )

                    print(
                        "Username:",
                        stored_username
                    )

                    print(
                        "=" * 60
                    )

                    return redirect(
                        url_for("home")
                    )

            print()
            print(
                "LOGIN FAILED"
            )

            return render_template(
                "login.html",
                mode="login",
                error="Invalid username or password.",
                username=username
            )

        except Exception as e:

            print()
            print(
                "LOGIN ERROR:"
            )

            print(
                str(e)
            )

            if connection:

                try:
                    connection.close()

                except:
                    pass

            return render_template(
                "login.html",
                mode="login",
                error=(
                    "Unable to log in right now. "
                    "Please try again."
                ),
                username=username
            )

    return render_template(
        "login.html",
        mode="login"
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

    print()
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

    if not is_logged_in():

        return redirect(
            url_for("login")
        )

    return render_template(
        "index.html",
        plant_names=plant_names,
        username=session.get("username"),
        history_page=False,
        health_score=None,
        health_status=None,
        health_healthy=None,
        health_attention=None,
        health_critical=None,
        environment=None
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    if not is_logged_in():

        return redirect(
            url_for("login")
        )

    user_history = get_user_history()

    return render_template(
        "index.html",
        history=user_history,
        history_page=True,
        plant_names=plant_names,
        username=session.get("username"),
        health_score=None,
        health_status=None,
        health_healthy=None,
        health_attention=None,
        health_critical=None,
        environment=None
    )


# ============================================================
# DELETE SELECTED HISTORY
# ============================================================

@app.route(
    "/delete-history/<int:history_id>",
    methods=["POST"]
)
def delete_history(history_id):

    if not is_logged_in():

        return redirect(
            url_for("login")
        )

    user_id = session.get(
        "user_id"
    )

    connection = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        if DATABASE_URL:

            cursor.execute(
                """
                DELETE FROM history
                WHERE id = %s
                AND user_id = %s
                """,
                (
                    history_id,
                    user_id
                )
            )

        else:

            cursor.execute(
                """
                DELETE FROM history
                WHERE id = ?
                AND user_id = ?
                """,
                (
                    history_id,
                    user_id
                )
            )

        connection.commit()

        cursor.close()
        connection.close()

        print()
        print(
            "HISTORY DELETED:",
            history_id
        )

    except Exception as e:

        print()
        print(
            "DELETE HISTORY ERROR:"
        )

        print(
            str(e)
        )

        if connection:

            try:

                connection.rollback()
                connection.close()

            except:
                pass

    return redirect(
        url_for("history")
    )


# ============================================================
# DELETE ALL HISTORY
# ============================================================

@app.route(
    "/delete-all-history",
    methods=["POST"]
)
def delete_all_history():

    if not is_logged_in():

        return redirect(
            url_for("login")
        )

    user_id = session.get(
        "user_id"
    )

    connection = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        if DATABASE_URL:

            cursor.execute(
                """
                DELETE FROM history
                WHERE user_id = %s
                """,
                (user_id,)
            )

        else:

            cursor.execute(
                """
                DELETE FROM history
                WHERE user_id = ?
                """,
                (user_id,)
            )

        connection.commit()

        cursor.close()
        connection.close()

        print()
        print(
            "ALL HISTORY DELETED FOR USER:",
            session.get("username")
        )

    except Exception as e:

        print()
        print(
            "DELETE ALL HISTORY ERROR:"
        )

        print(
            str(e)
        )

        if connection:

            try:

                connection.rollback()
                connection.close()

            except:
                pass

    return redirect(
        url_for("history")
    )


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if not is_logged_in():

        return redirect(
            url_for("login")
        )

    print()
    print(
        "=" * 60
    )

    print(
        "NEW IMAGE RECEIVED"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # CHECK IMAGE FIELD
    # ========================================================

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
                "IMAGE REJECTED - "
                "DOES NOT LOOK LIKE A PLANT"
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
        # MODEL PREDICTION (TFLite)
        # ====================================================

        interpreter.set_tensor(
            input_details[0]["index"],
            image_array.astype(
                input_details[0]["dtype"]
            )
        )

        interpreter.invoke()

        predictions = interpreter.get_tensor(
            output_details[0]["index"]
        )

        probabilities = predictions[0]

        # ====================================================
        # SAFETY CHECK
        # ====================================================

        if (
            len(probabilities)
            !=
            len(class_names)
        ):

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
        print(
            "TOP 5 AI PREDICTIONS"
        )

        print(
            "-" * 60
        )

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            print(
                rank,
                ".",
                class_names[
                    int(index)
                ],
                "->",
                round(
                    float(
                        probabilities[index]
                        * 100
                    ),
                    2
                ),
                "%"
            )

        print(
            "-" * 60
        )

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
            round(
                confidence,
                2
            ),
            "%"
        )

        print(
            "TOP-2 MARGIN:",
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

            print(
                "IMAGE REJECTED - "
                "UNCERTAIN PREDICTION"
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
                .replace(
                    "_",
                    " "
                )
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
        # AI HEALTH SCORE
        # ====================================================

        health_data = calculate_health_score(
            disease,
            confidence,
            severity
        )

        # ====================================================
        # RECOMMENDED ENVIRONMENT
        # ====================================================

        environment = environment_info.get(
            plant,
            {
                "sunlight": "6–8h",
                "soil_moisture": "Moderate",
                "humidity": "50–70%",
                "temperature": "18–28°C"
            }
        )

        # ====================================================
        # SUCCESS LOG
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

        print(
            "Health Score:",
            health_data["score"],
            "%"
        )

        print(
            "Health Status:",
            health_data["status"]
        )

        print(
            "User:",
            session.get(
                "username",
                "Unknown"
            )
        )

        print(
            "=" * 60
        )

        # ====================================================
        # SAVE RESULT TO HISTORY
        # ====================================================

        connection = None

        try:

            connection = get_db_connection()

            cursor = connection.cursor()

            user_id = session.get(
                "user_id"
            )

            if DATABASE_URL:

                cursor.execute(
                    """
                    INSERT INTO history
                    (
                        user_id,
                        plant,
                        disease,
                        confidence,
                        severity,
                        status,
                        care
                    )
                    VALUES
                    (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    """,
                    (
                        user_id,
                        plant,
                        disease,
                        confidence,
                        severity,
                        status,
                        care
                    )
                )

            else:

                cursor.execute(
                    """
                    INSERT INTO history
                    (
                        user_id,
                        plant,
                        disease,
                        confidence,
                        severity,
                        status,
                        care
                    )
                    VALUES
                    (
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        user_id,
                        plant,
                        disease,
                        confidence,
                        severity,
                        status,
                        care
                    )
                )

            connection.commit()

            cursor.close()

            connection.close()

            print(
                "Prediction saved to history."
            )

        except Exception as history_error:

            print()
            print(
                "HISTORY SAVE ERROR:"
            )

            print(
                str(history_error)
            )

            if connection:

                try:

                    connection.rollback()
                    connection.close()

                except:
                    pass

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

            plant_names=plant_names,

            username=session.get(
                "username"
            ),

            history_page=False,

            # =================================================
            # DYNAMIC HEALTH SCORE
            # =================================================

            health_score=health_data["score"],

            health_status=health_data["status"],

            health_healthy=health_data["healthy"],

            health_attention=health_data["attention"],

            health_critical=health_data["critical"],

            # =================================================
            # RECOMMENDED ENVIRONMENT
            # =================================================

            environment=environment

        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

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
            str(e)
        )

        print(
            "=" * 60
        )

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

        "model":
            "Plant Disease MobileNetV2 (TFLite)",

        "classes":
            len(class_names),

        "plants":
            len(plant_names),

        "database":
            "PostgreSQL"
            if DATABASE_URL
            else "SQLite",

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
    print(
        "=" * 60
    )

    print(
        "AI PLANT CARE SYSTEM"
    )

    print(
        "=" * 60
    )

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
        "Database:",
        "PostgreSQL"
        if DATABASE_URL
        else "SQLite"
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
    print(
        "LOGIN:"
    )

    print(
        "http://127.0.0.1:5000/login"
    )

    print()
    print(
        "REGISTER:"
    )

    print(
        "http://127.0.0.1:5000/register"
    )

    print()
    print(
        "HISTORY:"
    )

    print(
        "http://127.0.0.1:5000/history"
    )

    print()
    print(
        "SERVER STARTING..."
    )

    print(
        "=" * 60
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )
