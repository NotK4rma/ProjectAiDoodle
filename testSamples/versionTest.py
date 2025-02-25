
# import tensorflow as tf
# from tf_keras.models import load_model  # Or from tensorflow.keras.models if that works

# print("TensorFlow version:", tf.__version__)  # Print the version

# try:
#     model = load_model("K:\study\projectIA\doodleNet-model.h5")
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")


import h5py

with h5py.File("K:\study\projectIA\doodleNet-model.h5", "r") as f:
    print(f.attrs.get("keras_version"))
    print(f.attrs.get("backend"))
