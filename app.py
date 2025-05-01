from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)
interpreter = tf.lite.Interpreter(model_path="EfficientNetB3_skin_burn_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['image']
    img = Image.open(file.stream).resize((240, 240))  # Resize حسب الموديل
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run()
