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

# Full list of dermatology hospitals in Egypt
hospitals_list = [
    {"name": "مستشفى الحوض المرصود", "address": "السيدة زينب، القاهرة", "phone": "0223652367"},
    {"name": "مستشفى الأمراض الجلدية جامعة القاهرة", "address": "القصر العيني، القاهرة", "phone": "0223681211"},
    {"name": "مستشفى الجلدية جامعة عين شمس", "address": "العباسية، القاهرة", "phone": "0224824131"},
    {"name": "مستشفى جمال عبد الناصر", "address": "سيدي جابر، الإسكندرية", "phone": "033917207"},
    {"name": "مركز الأمراض الجلدية بالمنصورة", "address": "شارع الجمهورية، الدقهلية", "phone": "0502200133"}
]

# Instructions dictionary with first aid, hospitals, and emergency number
instructions = {
    "No Skin burn": {
        "first_aid": "لا يوجد حرق ظاهر في الصورة. لو عندك أعراض أو ألم، يُفضّل زيارة طبيب.",
        "hospitals": [],  # No hospitals needed for no burn
        "emergency_number": "123"
    },
    "1st degree": {
        "first_aid": """
✅ يجب فعله:
- ضع المنطقة المصابة تحت ماء بارد (ليس ثلج) لمدة 10-15 دقيقة.
- استخدم كريم مهدئ مثل ألوفيرا أو بانثينول.
- غطِ الحرق بشاش معقم فضفاض إذا لزم الأمر.
- تناول مسكنات مثل الباراسيتامول إذا كان هناك ألم.

❌ يجب تجنبه:
- لا تستخدم الثلج مباشرة على الحرق.
- لا تضع معجون أسنان، زبدة، أو وصفات شعبية.
- لا تفتح أي فقاعات إذا ظهرت.
- لا تفرك المنطقة المصابة.
""",
        "hospitals": hospitals_list,  # Full list of hospitals
        "emergency_number": "123"
    },
    "2nd degree": {
        "first_aid": """
✅ يجب فعله:
- اغسل المنطقة بماء بارد لمدة 15-30 دقيقة.
- لا تفتح الفقاعات إذا ظهرت.
- ضع كريم مضاد حيوي مثل سيلفر سلفاديازين (بعد استشارة طبيب) وغطِ الحرق بشاش معقم.
- تناول مسكنات للألم واشرب كميات كافية من الماء.
- استشر طبيبًا إذا كان الحرق كبيرًا أو في منطقة حساسة (مثل الوجه أو اليدين).

❌ يجب تجنبه:
- لا تفقع الفقاعات لأن ذلك يزيد خطر العدوى.
- لا تستخدم كريمات معطرة أو مواد دهنية مثل الزبدة.
- لا تلصق الضمادات بقوة على الحرق.
- لا تترك الحرق مكشوفًا إذا كان مفتوحًا.
""",
        "hospitals": hospitals_list,  # Full list of hospitals
        "emergency_number": "123"
    },
    "3rd degree": {
        "first_aid": """
❗ حالة طارئة - اطلب الإسعاف فورًا (رقم 123).

✅ يجب فعله:
- أبعد المصاب عن مصدر الحرق.
- غطِ الحرق بقطعة قماش نظيفة أو شاش معقم.
- ارفع المنطقة المصابة إذا أمكن لتقليل التورم.
- راقب التنفس والنبض حتى وصول الإسعاف.
- حافظ على دفء المصاب.

❌ يجب تجنبه:
- لا تضع ماء أو أي كريمات على الحرق.
- لا تحاول إزالة الملابس الملتصقة بالجلد.
- لا تعطِ المصاب طعامًا أو شرابًا إذا كان فاقدًا للوعي.
- لا تحاول تنظيف الحرق.
""",
        "hospitals": hospitals_list,  # Full list of hospitals
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
            "hospitals": info["hospitals"],
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