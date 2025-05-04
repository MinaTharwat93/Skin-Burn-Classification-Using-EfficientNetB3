import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import tensorflow as tf
from io import BytesIO
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Skin Burn Classification API")

# Constants (match notebook)
IMG_HEIGHT = 240
IMG_WIDTH = 240
CLASS_NAMES = ['No Skin burn', '1st degree', '2nd degree', '3rd degree']
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path="EfficientNetB3_skin_burn_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("TFLite model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load TFLite model: {str(e)}")
    raise

def preprocess_image(image: Image.Image):
    """Preprocess image for TFLite model."""
    try:
        image = image.convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
        image_array = np.array(image, dtype=np.float32)
        image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
        
        # Quantize to INT8
        input_scale = input_details[0]['quantization'][0]
        input_zero_point = input_details[0]['quantization'][1]
        if input_scale != 0:
            image_array = (image_array / input_scale) + input_zero_point
        image_array = np.clip(image_array, -128, 127).astype(np.int8)
        
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

async def predict_image(file: UploadFile):
    """Shared prediction logic for POST and GET requests."""
    try:
        # Check file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in SUPPORTED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}"
            )

        # Read and process image
        contents = await file.read()
        try:
            image = Image.open(BytesIO(contents))
        except Exception as e:
            logger.error(f"Invalid or corrupted image file: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

        image_array = preprocess_image(image)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize output
        output_scale = output_details[0]['quantization'][0]
        output_zero_point = output_details[0]['quantization'][1]
        dequantized_output = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        # Get prediction
        predicted_class_idx = np.argmax(dequantized_output[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        probabilities = dequantized_output[0].tolist()
        
        # Prepare response
        response = {
            "predicted_class": predicted_class,
            "probabilities": {
                CLASS_NAMES[i]: prob for i, prob in enumerate(probabilities)
            }
        }
        logger.info(f"Prediction: {response}")
        return response
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict")
async def predict_post(file: UploadFile = File(...)):
    """Predict skin burn class from uploaded image via POST."""
    return await predict_image(file)

@app.get("/predict")
async def predict_get(file: UploadFile = File(...)):
    """Predict skin burn class from uploaded image via GET."""
    return await predict_image(file)