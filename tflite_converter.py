import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("mouseDetectorModel/my_model/saved_model") # path to the SavedModel directory
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.

]

converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

# Save the model.
with open('mouse_model.tflite', 'wb') as f:
  f.write(tflite_model)

print("done!")