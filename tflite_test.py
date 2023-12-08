import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="mouse_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()[0]["index"]
print(input_details)
output_details = interpreter.get_output_details()[0]["index"]
print(output_details)