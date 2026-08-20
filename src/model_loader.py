import tensorflow as tf
import os

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    return model