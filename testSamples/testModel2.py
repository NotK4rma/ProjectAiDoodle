from tf_keras.models import model_from_json
import numpy as np
import PIL as Image


with open(r"K:\study\projectIA\NewModel\model.json"       , "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)



with open(r"K:\study\projectIA\NewModel\group1-shard1of1.bin", "rb") as bin_file:
    weights = np.load(bin_file, allow_pickle=True)
model.set_weights(weights)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

with open(r"K:\study\projectIA\NewModel\class_names.txt", "r") as f:
    CLASS_LABELS = [line.strip() for line in f]



img = Image.open(r"K:/study/projectIA/testSamples/pic6.png")
img = img.resize((28,28))
img = img.convert("L")


img_array = np.array(img)/255
img_array = img_array.reshape(1, 28, 28, 1)
predictions = model.predict(img_array)



print("Predicted class:", CLASS_LABELS[np.argmax(predictions)])
