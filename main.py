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

# Instructions dictionary with first aid (in English) and emergency number
instructions = {
    "No Skin burn": {
        "first_aid": "No visible burn detected in the image. If you experience symptoms or pain, consider consulting a doctor.",
        "emergency_number": "123"
    },
    "1st degree": {
        "first_aid": """
✅ What to do:
- Place the affected area under cool (not ice-cold) running water for 10-15 minutes.
- Apply a soothing cream like aloe vera or panthenol.
- Cover the burn with a sterile, loose gauze if needed.
- Take over-the-counter pain relievers like paracetamol if there is pain.
"""
"""
❌ What to avoid:
- Do not apply ice directly to the burn.
- Do not use toothpaste, butter, or home remedies.
- Do not pop any blisters if they appear.
- Do not rub the affected area.
""",
        "emergency_number": "123"
    },
    "2nd degree": {
        "first_aid": """
✅ What to do:
- Rinse the area with cool running water for 15-30 minutes.
- Do not pop any blisters if they appear.
- Apply an antibiotic cream like silver sulfadiazine (after consulting a doctor) and cover the burn with sterile gauze.
- Take pain relievers and drink plenty of water.
- Seek medical advice if the burn is large or in a sensitive area (e.g., face or hands).
"""
"""
❌ What to avoid:
- Do not pop blisters, as this increases the risk of infection.
- Do not use scented creams or greasy substances like butter.
- Do not apply adhesive bandages tightly to the burn.
- Do not leave the burn exposed if it is open.
""",
        "emergency_number": "123"
    },
    "3rd degree": {
        "first_aid": """
❗ Emergency - Call an ambulance immediately (number: 123).

✅ What to do:
- Remove the person from the source of the burn.
- Cover the burn with a clean cloth or sterile gauze.
- Elevate the affected area if possible to reduce swelling.
- Monitor breathing and pulse until help arrives.
- Keep the person warm.
"""
"""
❌ What to avoid:
- Do not apply water or any creams to the burn.
- Do not attempt to remove clothing stuck to the skin.
- Do not give food or drink if the person is unconscious.
- Do not try to clean the burn.
""",
        "emergency_number": "123"
    }
}

def preprocess_image(image: Image.Image):
    try:
        image = image.convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
        image_array = np.array(image, dtype=np.float32)
        image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)

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
    try:
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in SUPPORTED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}"
            )

        contents = await file.read()
        try:
            image = Image.open(BytesIO(contents))
        except Exception as e:
            logger.error(f"Invalid or corrupted image file: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

        image_array = preprocess_image(image)

        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        output_scale = output_details[0]['quantization'][0]
        output_zero_point = output_details[0]['quantization'][1]
        dequantized_output = (output_data.astype(np.float32) - output_zero_point) * output_scale

        predicted_class_idx = np.argmax(dequantized_output[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        probabilities = dequantized_output[0].tolist()

        info = instructions[predicted_class]

        response = {
            "predicted_class": predicted_class,
            "probabilities": {
                CLASS_NAMES[i]: prob for i, prob in enumerate(probabilities)
            },
            "first_aid": info["first_aid"],
            "emergency_number": info["emergency_number"]
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
    return await predict_image(file)

@app.get("/predict")
async def predict_get(file: UploadFile = File(...)):
    return await predict_image(file)