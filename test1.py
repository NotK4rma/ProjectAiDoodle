from flask import Flask, render_template, request, jsonify
from tf_keras.models import load_model
import numpy as np
import base64
from PIL import Image
import io
import cv2


app = Flask(__name__)


model = load_model(r"K:\study\projectIA\doodleNet-model.h5")


with open(r"K:\study\projectIA\class_names.txt", "r") as f:
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
    img = Image.open(io.BytesIO(image_data))

    img = img.resize((28,28))
    img=img.convert("L")


    img_array = (255-np.array(img))/255
    img_array = img_array.reshape(1, 28, 28, 1)


    #print(img)


    predictions = model.predict(img_array)
    # print(predictions)
    predicted_class = CLASS_LABELS[np.argmax(predictions)]
    # print(predicted_class)

    return jsonify({"prediction": predicted_class}) 

if __name__ == "__main__":
    app.run(debug=True)
