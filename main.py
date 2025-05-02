from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
from tensorflow.lite.python.interpreter import Interpreter

app = FastAPI()

# Load the TensorFlow Lite model
interpreter = Interpreter(model_path="EfficientNetB3_skin_burn_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Log input and output details for debugging
print("Input details:", input_details)
print("Output details:", output_details)

# Class labels for burn severity
labels = ["No Burn", "First Degree", "Second Degree"]

# Function to preprocess the image for INT8 input
def preprocess_image(image: Image.Image):
    # Resize the image to the size expected by the model (240x240)
    image = image.resize((240, 240))
    # Convert the image to a NumPy array (values in [0, 255])
    image_array = np.array(image, dtype=np.float32)
    # Normalize to [0, 1]
    image_array = image_array / 255.0

    # Get quantization parameters from input details
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]

    # Quantize the image to INT8
    if input_scale != 0:  # Avoid division by zero
        image_array = (image_array / input_scale) + input_zero_point
    image_array = np.clip(image_array, -128, 127).astype(np.int8)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess the image
        input_data = preprocess_image(image)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]["index"])
        
        # Dequantize the output
        output_scale = output_details[0]['quantization'][0]
        output_zero_point = output_details[0]['quantization'][1]
        dequantized_output = (output_data.astype(np.float32) - output_zero_point) * output_scale

        # Convert to probabilities (if softmax was applied during training)
        dequantized_output = np.exp(dequantized_output) / np.sum(np.exp(dequantized_output), axis=-1, keepdims=True)
        
        # Convert to list for JSON response
        prediction = dequantized_output.tolist()
        predicted_label = labels[np.argmax(prediction[0])]  # Get the predicted class
        
        return JSONResponse(content={"prediction": prediction, "label": predicted_label})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)