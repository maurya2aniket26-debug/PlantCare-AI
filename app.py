```python
import os

# =========================================================
# IMPORTANT: CPU ONLY FOR RENDER
# =========================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import base64
import uuid
import gc

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


# =========================================================
# 1. LIMIT TENSORFLOW CPU THREADS
# =========================================================

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


# =========================================================
# 2. FLASK APP
# =========================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# =========================================================
# 3. PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plant_disease_mobilenet_finetuned.keras"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# 4. CLASS NAMES
# =========================================================

class_names = [
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


# =========================================================
# 5. DISEASE INFORMATION
# =========================================================

disease_info = {

    "Apple___Apple_scab": (
        "Apple",
        "Apple Scab",
        "Remove affected leaves and fruit, improve air circulation, and follow appropriate disease-management guidance."
    ),

    "Apple___Black_rot": (
        "Apple",
        "Black Rot",
        "Remove infected plant material and maintain good orchard sanitation."
    ),

    "Apple___Cedar_apple_rust": (
        "Apple",
        "Cedar Apple Rust",
        "Remove severely affected leaves and follow appropriate disease-management guidance."
    ),

    "Apple___healthy": (
        "Apple",
        "Healthy",
        "The leaf appears healthy. Continue regular watering, nutrition, and monitoring."
    ),

    "Blueberry___healthy": (
        "Blueberry",
        "Healthy",
        "The plant appears healthy. Continue normal care and monitoring."
    ),

    "Cherry_(including_sour)___Powdery_mildew": (
        "Cherry",
        "Powdery Mildew",
        "Improve air circulation, reduce excessive humidity, and follow appropriate disease-management guidance."
    ),

    "Cherry_(including_sour)___healthy": (
        "Cherry",
        "Healthy",
        "The leaf appears healthy. Continue regular plant care."
    ),

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (
        "Corn",
        "Cercospora / Gray Leaf Spot",
        "Maintain field sanitation, improve crop management, and follow appropriate disease-management guidance."
    ),

    "Corn_(maize)___Common_rust_": (
        "Corn",
        "Common Rust",
        "Monitor the crop and follow appropriate disease-management practices."
    ),

    "Corn_(maize)___Northern_Leaf_Blight": (
        "Corn",
        "Northern Leaf Blight",
        "Use resistant varieties where available and follow appropriate disease-management practices."
    ),

    "Corn_(maize)___healthy": (
        "Corn",
        "Healthy",
        "The plant appears healthy. Continue normal crop management."
    ),

    "Grape___Black_rot": (
        "Grape",
        "Black Rot",
        "Remove infected leaves and fruit, maintain sanitation, and follow appropriate disease-management guidance."
    ),

    "Grape___Esca_(Black_Measles)": (
        "Grape",
        "Esca (Black Measles)",
        "Remove severely affected material and consult a local plant-disease expert for serious cases."
    ),

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (
        "Grape",
        "Leaf Blight",
        "Remove affected leaves, improve air circulation, and follow appropriate disease-management guidance."
    ),

    "Grape___healthy": (
        "Grape",
        "Healthy",
        "The plant appears healthy. Continue normal vineyard care."
    ),

    "Orange___Haunglongbing_(Citrus_greening)": (
        "Orange",
        "Huanglongbing (Citrus Greening)",
        "Consult a local agricultural expert and follow local disease-management guidance."
    ),

    "Peach___Bacterial_spot": (
        "Peach",
        "Bacterial Spot",
        "Remove severely affected material and maintain good orchard sanitation."
    ),

    "Peach___healthy": (
        "Peach",
        "Healthy",
        "The plant appears healthy. Continue normal care."
    ),

    "Pepper,_bell___Bacterial_spot": (
        "Bell Pepper",
        "Bacterial Spot",
        "Remove affected material, avoid overhead watering, and maintain good plant sanitation."
    ),

    "Pepper,_bell___healthy": (
        "Bell Pepper",
        "Healthy",
        "The plant appears healthy. Continue regular plant care."
    ),

    "Potato___Early_blight": (
        "Potato",
        "Early Blight",
        "Remove affected leaves where practical, avoid prolonged leaf wetness, and follow appropriate disease-management guidance."
    ),

    "Potato___Late_blight": (
        "Potato",
        "Late Blight",
        "Remove severely affected leaves, avoid prolonged leaf wetness, and seek appropriate disease-management guidance."
    ),

    "Potato___healthy": (
        "Potato",
        "Healthy",
        "The plant appears healthy. Continue normal crop management."
    ),

    "Raspberry___healthy": (
        "Raspberry",
        "Healthy",
        "The plant appears healthy. Continue regular care."
    ),

    "Soybean___healthy": (
        "Soybean",
        "Healthy",
        "The plant appears healthy. Continue normal crop monitoring."
    ),

    "Squash___Powdery_mildew": (
        "Squash",
        "Powdery Mildew",
        "Improve air circulation, reduce excessive humidity, and follow appropriate disease-management guidance."
    ),

    "Strawberry___Leaf_scorch": (
        "Strawberry",
        "Leaf Scorch",
        "Remove severely affected leaves and maintain appropriate watering and growing conditions."
    ),

    "Strawberry___healthy": (
        "Strawberry",
        "Healthy",
        "The plant appears healthy. Continue regular plant care."
    ),

    "Tomato___Bacterial_spot": (
        "Tomato",
        "Bacterial Spot",
        "Remove affected leaves, avoid overhead watering, and maintain good plant sanitation."
    ),

    "Tomato___Early_blight": (
        "Tomato",
        "Early Blight",
        "Remove affected leaves, avoid overhead watering, improve air circulation, and follow appropriate disease-management guidance."
    ),

    "Tomato___Late_blight": (
        "Tomato",
        "Late Blight",
        "Remove severely affected leaves, avoid prolonged leaf wetness, and seek appropriate disease-management guidance."
    ),

    "Tomato___Leaf_Mold": (
        "Tomato",
        "Leaf Mold",
        "Improve ventilation, reduce humidity, avoid wetting leaves, and remove severely affected leaves."
    ),

    "Tomato___Septoria_leaf_spot": (
        "Tomato",
        "Septoria Leaf Spot",
        "Remove affected leaves, avoid overhead watering, and maintain good garden sanitation."
    ),

    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Tomato",
        "Spider Mites",
        "Inspect the underside of leaves and use suitable pest-management methods."
    ),

    "Tomato___Target_Spot": (
        "Tomato",
        "Target Spot",
        "Remove affected leaves, improve air circulation, and avoid prolonged leaf wetness."
    ),

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Tomato",
        "Tomato Yellow Leaf Curl Virus",
        "Manage whitefly vectors and follow local agricultural guidance."
    ),

    "Tomato___Tomato_mosaic_virus": (
        "Tomato",
        "Tomato Mosaic Virus",
        "Remove infected plants where appropriate and sanitize tools to reduce disease spread."
    ),

    "Tomato___healthy": (
        "Tomato",
        "Healthy",
        "The tomato leaf appears healthy. Continue regular watering, nutrition, and monitoring."
    )
}


# =========================================================
# 6. LOAD MODEL
# =========================================================

print("Loading fine-tuned MobileNetV2 model...")

try:

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print(
        "Fine-tuned MobileNetV2 model loaded successfully!"
    )

except Exception as e:

    print("MODEL LOAD ERROR:", e)

    model = None


# =========================================================
# 7. IMAGE PREPARATION
# =========================================================

def prepare_image(image_path):

    img = tf.keras.utils.load_img(
        image_path,
        target_size=(160, 160)
    )

    img_array = tf.keras.utils.img_to_array(
        img
    )

    img_array = tf.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# =========================================================
# 8. HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# 9. PREDICT
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    image_path = None
    filename = None

    # -----------------------------------------------------
    # UPLOAD IMAGE
    # -----------------------------------------------------

    if (
        "image" in request.files
        and request.files["image"].filename != ""
    ):

        file = request.files["image"]

        original_name = secure_filename(
            file.filename
        )

        extension = os.path.splitext(
            original_name
        )[1].lower()

        if extension not in [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        ]:

            return render_template(
                "index.html",
                error="Please upload JPG, JPEG, PNG or WEBP image."
            )

        filename = (
            "uploaded_"
            + uuid.uuid4().hex
            + extension
        )

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(
            image_path
        )


    # -----------------------------------------------------
    # CAMERA IMAGE
    # -----------------------------------------------------

    elif request.form.get(
        "camera_image_data"
    ):

        camera_data = request.form.get(
            "camera_image_data"
        )

        try:

            if "," in camera_data:

                camera_data = camera_data.split(
                    ",",
                    1
                )[1]

            image_bytes = base64.b64decode(
                camera_data
            )

            filename = (
                "camera_"
                + uuid.uuid4().hex
                + ".jpg"
            )

            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            with open(
                image_path,
                "wb"
            ) as image_file:

                image_file.write(
                    image_bytes
                )

        except Exception as e:

            print(
                "Camera image error:",
                e
            )

            return render_template(
                "index.html",
                error="Could not process the camera photo."
            )

    else:

        return render_template(
            "index.html",
            error="Please upload an image or take a photo."
        )


    # -----------------------------------------------------
    # MODEL CHECK
    # -----------------------------------------------------

    if model is None:

        return render_template(
            "index.html",
            error="The AI model could not be loaded."
        )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    try:

        print("Preparing image...")

        img_array = prepare_image(
            image_path
        )

        print("Running AI prediction...")

        # Faster/lighter than model.predict()
        predictions = model.predict_on_batch(
            img_array
        )[0]

        predictions = np.asarray(
            predictions,
            dtype=np.float32
        )

        print("Prediction completed.")


        # -------------------------------------------------
        # TOP 3
        # -------------------------------------------------

        top_indices = np.argsort(
            predictions
        )[-3:][::-1]

        top_predictions = []

        for index in top_indices:

            top_predictions.append(
                {
                    "class": class_names[index],

                    "confidence": round(
                        float(
                            predictions[index] * 100
                        ),
                        2
                    )
                }
            )


        # -------------------------------------------------
        # BEST PREDICTION
        # -------------------------------------------------

        predicted_index = int(
            top_indices[0]
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = float(
            predictions[predicted_index] * 100
        )


        # -------------------------------------------------
        # UNCERTAIN
        # -------------------------------------------------

        UNCERTAIN_THRESHOLD = 55.0

        if confidence < UNCERTAIN_THRESHOLD:

            plant = ""

            disease = ""

            solution = (
                "The image could not be identified "
                "with sufficient confidence. Please "
                "capture a clear, close-up image of "
                "a single plant leaf in good lighting, "
                "with minimal background and the "
                "affected area clearly visible."
            )

            confidence_message = (
                "Keep the leaf centered and avoid "
                "blurry or distant images."
            )

            prediction_status = "uncertain"


        # -------------------------------------------------
        # NORMAL PREDICTION
        # -------------------------------------------------

        else:

            if predicted_class in disease_info:

                plant, disease, solution = disease_info[
                    predicted_class
                ]

            else:

                plant = "Unknown"

                disease = predicted_class

                solution = (
                    "Please consult a plant-disease expert."
                )


            if confidence >= 80:

                confidence_message = (
                    "The model is highly confident "
                    "in this prediction."
                )

            elif confidence >= 70:

                confidence_message = (
                    "The model is reasonably confident "
                    "in this prediction."
                )

            else:

                confidence_message = (
                    "The model has moderate confidence. "
                    "Try a clearer close-up image for a "
                    "more reliable prediction."
                )

            prediction_status = "prediction"


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        result = render_template(

            "index.html",

            prediction=disease,

            plant=plant,

            confidence=round(
                confidence,
                2
            ),

            solution=solution,

            confidence_message=confidence_message,

            prediction_status=prediction_status,

            top_predictions=top_predictions,

            image_url=(
                "/static/uploads/"
                + filename
            )
        )


        # -------------------------------------------------
        # CLEAN MEMORY
        # -------------------------------------------------

        del img_array
        del predictions

        gc.collect()

        return result


    except Exception as e:

        print(
            "Prediction error:",
            repr(e)
        )

        return render_template(
            "index.html",
            error="Unable to analyze this image. Please try again with a clear leaf photo."
        )


# =========================================================
# 10. RUN
# =========================================================

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
```
