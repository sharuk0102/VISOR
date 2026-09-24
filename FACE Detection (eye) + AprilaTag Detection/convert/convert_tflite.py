import os
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope

def relu6(x):
    return tf.nn.relu6(x)

def convert_h5_to_tflite():
    model_path = "model/yolov3_lite_model.h5"
    tflite_path = "convert/yolov3_lite.tflite"
    
    if not os.path.exists(model_path):
        print(f"Error: Keras model not found at {model_path}! Run training first.")
        return
        
    print(f"Loading Keras model from {model_path}...")
    # Wrap in custom_object_scope to load relu6 custom activation correctly
    with custom_object_scope({'relu6': relu6}):
        model = tf.keras.models.load_model(model_path)
        
    print("Converting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Configure converter parameters
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # Create directory and save
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Saved TFLite model to {tflite_path}")
    print(f"Size: {len(tflite_model) / 1024:.2f} KB")

if __name__ == "__main__":
    convert_h5_to_tflite()
