from flask import Flask, render_template, request, jsonify
from tf_keras.models import load_model
import numpy as np
import base64
from PIL import Image
import io


app = Flask(__name__)


model = load_model("doodleNet-model.h5")


with open("categories.txt", "r") as f:
    CLASS_LABELS = [line.strip() for line in f]

@app.route("/")
def index():
    return render_template("index.html")  

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    
    image_data = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(image_data)).convert("L")  
    image = image.resize((28, 28))  
    image_array = np.array(image) / 255.0  
    image_array = image_array.reshape(1, 28, 28, 1) 

    
    predictions = model.predict(image_array)
    predicted_class = CLASS_LABELS[np.argmax(predictions)]
    

    return jsonify({"prediction": predicted_class}) 

if __name__ == "__main__":
    app.run(debug=True)
