#Chech DoodleNet by yining1023 On GitHub for creds + tips
import numpy as np
from PIL import Image
from tf_keras.models import load_model

model = load_model("K:\study\projectIA\doodleNet-model.h5")


with open(r"K:\study\projectIA\class_names.txt", "r") as f:
    CLASS_LABELS = [line.strip() for line in f]

img = Image.open("K:/study/projectIA/testSamples/pic6.png")
img = img.resize((28,28))
img = img.convert("L")


img_array = (255-np.array(img))/255
img_array = img_array.reshape(1, 28, 28, 1)


predictions = model.predict(img_array)


print("Predicted class:", CLASS_LABELS[np.argmax(predictions)])