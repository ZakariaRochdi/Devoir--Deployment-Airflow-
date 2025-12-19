import os
import numpy as np
import joblib

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# ------------------- VGG16 -------------------
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16

# ------------------- GPT-2 -------------------
from transformers import pipeline

# ------------------- Flask Config -------------------
app = Flask(__name__)

UPLOAD_FOLDER = "./images/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

# ------------------- Load Models -------------------

# 1️⃣ VGG16
try:
    vgg_model = VGG16()
    print("✅ VGG16 chargé")
except Exception as e:
    print("❌ Erreur VGG16:", e)
    vgg_model = None

# 2️⃣ Régression (joblib)
regression_model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/model.joblib")

try:
    regression_model = joblib.load(MODEL_PATH)
    print("✅ Modèle de régression chargé")
except Exception as e:
    print("❌ Erreur chargement modèle régression:", e)

# 3️⃣ GPT-2
try:
    text_generator = pipeline("text-generation", model="gpt2")
    print("✅ GPT-2 chargé")
except Exception as e:
    print("❌ Erreur GPT-2:", e)
    text_generator = None

# ------------------- Utils -------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_int(word):
    mapping = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        0: 0,
    }
    return mapping.get(word.lower(), 0)


# ------------------- Routes -------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------- Image Classification ----------
@app.route("/predict", methods=["POST"])
def predict():
    if "imagefile" not in request.files:
        return render_template("index.html", error="Aucun fichier sélectionné")

    imagefile = request.files["imagefile"]

    if imagefile.filename == "" or not allowed_file(imagefile.filename):
        return render_template("index.html", error="Fichier image invalide")

    filename = secure_filename(imagefile.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    imagefile.save(image_path)

    try:
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, *image.shape))
        image = preprocess_input(image)

        preds = vgg_model.predict(image)
        label = decode_predictions(preds)[0][0]
        result = f"{label[1]} ({label[2]*100:.2f}%)"

        os.remove(image_path)
        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", error=str(e))


# ---------- Regression ----------
@app.route("/regpredict", methods=["POST"])
def regpredict():
    if regression_model is None:
        return render_template("index.html", error="Modèle de régression non chargé")

    try:
        experience = convert_to_int(request.form.get("experience"))
        test_score = float(request.form.get("test_score"))
        interview_score = float(request.form.get("interview_score"))

        X = np.array([[experience, test_score, interview_score]])
        prediction = regression_model.predict(X)

        result = f"Salaire prédit : {prediction[0]:.2f}"
        return render_template("index.html", regression_prediction=result)

    except Exception as e:
        return render_template("index.html", error=f"Erreur régression: {e}")


# ---------- Text Generation ----------
@app.route("/textgen", methods=["POST"])
def textgen():
    if text_generator is None:
        return render_template("index.html", error="GPT-2 non chargé")

    prompt = request.form.get("prompt_text")

    try:
        output = text_generator(prompt, max_length=50, num_return_sequences=1)
        generated = output[0]["generated_text"]
        return render_template("index.html", textgen_result=generated)

    except Exception as e:
        return render_template("index.html", error=str(e))


# ------------------- Main -------------------
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=3000)
