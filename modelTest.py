from tf_keras.models import load_model
import numpy as np

model = load_model("quickdraw_model.h5")
sample_input = np.random.rand(1, 28, 28, 1)  # Random input for testing
predictions = model.predict(sample_input)
print("Sample predictions:", predictions)
