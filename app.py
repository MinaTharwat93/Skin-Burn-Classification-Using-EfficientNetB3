import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --- Page Config ---
st.set_page_config(page_title="Skin Burn Classification", page_icon="Fire", layout="wide")

# --- Navigation State ---
if 'page' not in st.session_state:
    st.session_state.page = 1

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

# --- 2D Design: Skin Burn Theme (أعلى الصفحة الأولى) ---
if st.session_state.page == 1:
    html_burn_design = """
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
          background: linear-gradient(135deg, #ff6b35, #ff4500);
          font-family: 'Segoe UI', sans-serif;
          color: white;
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          text-align: center;
          padding: 20px;
        }
        .container {
          max-width: 1100px;
          animation: fadeIn 1.5s ease-out;
        }
        .title {
          font-size: 3.8rem;
          font-weight: 900;
          margin-bottom: 10px;
          text-shadow: 0 6px 15px rgba(0,0,0,0.4);
          letter-spacing: 1.5px;
        }
        .subtitle {
          font-size: 1.6rem;
          margin-bottom: 35px;
          opacity: 0.95;
          font-weight: 500;
        }
        .flow {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 30px;
          flex-wrap: wrap;
          margin: 40px 0;
        }
        .node {
          background: rgba(255, 255, 255, 0.18);
          backdrop-filter: blur(12px);
          border-radius: 22px;
          padding: 22px 30px;
          min-width: 180px;
          text-align: center;
          box-shadow: 0 12px 28px rgba(0,0,0,0.3);
          transition: all 0.4s ease;
          border: 2px solid transparent;
        }
        .node:hover {
          transform: translateY(-12px);
          box-shadow: 0 20px 40px rgba(0,0,0,0.4);
          border-color: #ffd700;
        }
        .node img {
          width: 70px;
          height: 70px;
          margin-bottom: 14px;
          filter: drop-shadow(0 4px 8px rgba(0,0,0,0.4));
        }
        .node h3 {
          font-size: 1.3rem;
          margin-bottom: 6px;
          font-weight: 700;
        }
        .node p {
          font-size: 0.95rem;
          opacity: 0.9;
        }
        .arrow {
          font-size: 3rem;
          color: #ffd700;
          animation: pulse 2s infinite;
          font-weight: bold;
        }
        @keyframes pulse {
          0%,100% { opacity:0.7; transform: scale(1); }
          50% { opacity:1; transform: scale(1.15); }
        }
        .burn-levels {
          display: flex;
          justify-content: center;
          gap: 20px;
          margin-top: 40px;
          flex-wrap: wrap;
        }
        .level {
          background: rgba(0,0,0,0.3);
          padding: 14px 26px;
          border-radius: 50px;
          font-size: 1.05rem;
          font-weight: 600;
          backdrop-filter: blur(6px);
          transition: 0.35s;
          box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }
        .level:hover {
          background: #ff4500;
          color: white;
          transform: scale(1.08);
          box-shadow: 0 8px 20px rgba(255,69,0,0.5);
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-25px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 768px) {
          .title { font-size: 2.6rem; }
          .subtitle { font-size: 1.2rem; }
          .flow { flex-direction: column; gap: 25px; }
          .arrow { transform: rotate(90deg); }
          .node img { width: 60px; height: 60px; }
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1 class="title">Skin Burn Classification</h1>
        <p class="subtitle">AI-Powered Automatic Burn Detection</p>

        <div class="flow">
          <div class="node">
            <img src="https://cdn-icons-png.flaticon.com/512/2913/2913022.png" alt="Burned Skin">
            <h3>Burn Image</h3>
            <p>Skin Damage Input</p>
          </div>
          <div class="arrow">Right Arrow</div>
          <div class="node">
            <img src="https://cdn-icons-png.flaticon.com/512/1995/1995512.png" alt="AI Brain">
            <h3>EfficientNetB3</h3>
            <p>AI Classification</p>
          </div>
        </div>

        <div class="burn-levels">
          <div class="level">No Burn</div>
          <div class="level">1st-degree</div>
          <div class="level">2nd-degree</div>
          <div class="level">3rd-degree</div>
        </div>
      </div>
    </body>
    </html>
    """

    # عرض التصميم في أعلى الصفحة الأولى
    st.components.v1.html(html_burn_design, height=580, scrolling=False)

    # --- باقي المحتوى ---
    st.markdown("---")
    st.title("Skin Burn Classification using EfficientNetB3")

    st.header("Project Overview")
    st.write(
        """Burn assessment is a vital medical process — incorrect diagnosis can cause infections, deep tissue damage, or long-term scarring.
        However, access to burn specialists is often limited, especially in emergency or rural areas.

        This project aims to develop an AI-based system capable of automatically classifying skin burn images into four categories:
        **No Burn**, **1st-degree**, **2nd-degree**, and **3rd-degree**.
        Our system provides a fast, consistent, and objective diagnostic tool to support healthcare professionals."""
    )

    st.header("Project Goal")
    st.markdown(
        "- Evaluate and compare multiple deep learning models.\n"
        "- Achieve **high accuracy** and **strong generalization** across unseen data.\n"
        "- Deploy the final model in a **Streamlit-based interactive interface** for real-time testing."
    )

    st.header("Dataset")
    st.markdown(
        """The dataset was sourced from **Kaggle** and contains **47,683 clinical skin images**:
        - 18,181 → 1st-degree burns  
        - 17,093 → 2nd-degree burns  
        - 9,294 → 3rd-degree burns  
        - 3,115 → Normal skin  

        **Challenges:**  
        - Class imbalance  
        - Similar visual patterns between degrees  
        - Varying lighting and image quality"""
    )

    st.header("Image Preprocessing")
    st.markdown(
        "- **Resizing (240×240)** with padding to maintain proportions.\n"
        "- **Data Augmentation:** random flips, brightness & contrast changes.\n"
        "- **Class Weighting:** handle imbalance and improve fairness.\n\n"
        "These steps improved the model’s ability to learn and generalize effectively."
    )

    st.button("Next Right Arrow", on_click=next_page, use_container_width=True)

# --- PAGE 2 ---
elif st.session_state.page == 2:
    st.title("EfficientNetB3 Model & Deployment")
    # ... (باقي الكود كما هو، بدون تغيير)
    st.header("Why EfficientNetB3?")
    st.write(
        """After comparing several CNN architectures, **EfficientNetB3** achieved the best balance between accuracy, speed, and generalization.
        It uses **compound scaling**, balancing depth, width, and resolution. Pretrained on ImageNet, it adapts well to medical data."""
    )
    st.write("Higher accuracy with fewer parameters (compound scaling). Efficient and suitable for medical tasks. Pretrained on ImageNet, easy to adapt.")

    st.header("Difference from Other CNN Models")
    st.table({
        "Feature": ["Scaling Method", "Model Size vs. Accuracy", "Pretrained Availability", "Performance"],
        "EfficientNetB3": ["Compound scaling (width, depth, resolution)", "Smaller yet more accurate", "Yes (on ImageNet)", "High accuracy with less computation"],
        "Traditional CNN (e.g., VGG, ResNet)": ["Manual/deep stacking layers", "Larger models for similar performance", "Yes, but typically bulkier", "May require more resources"]
    })
    st.write("EfficientNetB3 is more balanced, combining efficiency and power—ideal for complex classification tasks on limited hardware.")

    st.header("Handling Photos and Colors")
    st.write("Images are resized to 240x240. Color augmentation techniques (like brightness, contrast, and flipping) are applied to make the model more robust. EfficientNetB3 expects RGB input and uses efficientnet.preprocess_input() to normalize pixel values.")

    st.header("Model Structure")
    st.markdown(
        "- Input Layer – Accepts 240x240x3 RGB images.\n"
        "- EfficientNetB3 Base – Pretrained on ImageNet (include_top=False).\n"
        "- GlobalAveragePooling2D – Reduces dimensions while retaining key features.\n"
        "- Dense Layer (128 units) – With ReLU activation & L2 regularization.\n"
        "- Batch Normalization – Stabilizes learning.\n"
        "- Dropout (0.6) – Prevents overfitting.\n"
        "- Output Layer – 4 units (softmax) for each skin burn class."
    )

    st.header("Implementation Details")
    st.markdown(
        """
        - **Data Augmentation:** Randomly flipping images horizontally and vertically. Randomly changing the brightness and contrast.
        - **Optimization & Regularization:** Adam optimizer was used for its adaptive learning rate capabilities. A learning rate schedule was implemented to decay the learning rate after an initial number of epochs for more stable convergence. L2 regularization was applied to the final dense layer to prevent overfitting. Class weights were computed and used during training to handle the imbalanced dataset, ensuring the model paid more attention to minority classes.
        - **Training Strategy:** Early stopping was used to halt training if validation performance did not improve, preventing the model from overfitting. Model checkpointing was enabled to save only the best version of the model based on its validation accuracy. Key training parameters: Batch size: 32, Image size: 240x240, Epochs: 10 (initial training) + 15 (fine-tuning).
        """
    )

    st.header("Model Architecture")
    st.markdown(
        "- Input: RGB images (240×240) normalized using `efficientnet.preprocess_input()`.\n"
        "- **EfficientNetB3 Base:** pretrained on ImageNet (without top layers).\n"
        "- **Global Average Pooling:** reduces dimensionality.\n"
        "- **Dense Layer (128 units):** learns burn-specific patterns.\n"
        "- **Batch Normalization + Dropout (0.6):** prevent overfitting.\n"
        "- **Output Layer:** 4 neurons (softmax) for each burn degree."
    )

    st.header("Training Strategy")
    st.markdown(
        "1. **Phase 1:** Train the classification head (EfficientNet frozen).\n"
        "2. **Phase 2:** Fine-tune the top 30 layers of EfficientNet.\n\n"
        "**Loss:** Categorical Cross-Entropy  \n        **Metrics:** Accuracy & AUC  \n        **Optimizer:** Adam  \n        **Callbacks:** EarlyStopping, ModelCheckpoint, LearningRateScheduler"
    )

    st.header("Results")
    st.table({
        "Metric": ["Training Accuracy", "Validation Accuracy", "Test Accuracy", "AUC Score"],
        "Result": ["95.38%", "95.41%", "95.74%", "0.9954"]
    })

    st.header("Deployment using Streamlit")
    st.write(
        """We deployed the trained EfficientNetB3 model using **Streamlit** for real-time interaction.

        - The model was converted to **TensorFlow Lite (.tflite)** format for faster inference.
        - A **Streamlit interface** allows users to upload images and view predictions instantly.
        - Confidence levels are visualized in a **bar chart**.
        - The app provides automatic **first-aid guidance** for each burn degree.

        **Why Streamlit?**  
        - Lightweight and fast  
        - No server setup required  
        - Easy to deploy locally or on Streamlit Cloud  
        - Intuitive UI for healthcare professionals and researchers"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.button("Left Arrow Back", on_click=prev_page, use_container_width=True)
    with col2:
        st.button("Next Right Arrow", on_click=next_page, use_container_width=True)

# --- PAGE 3 ---
elif st.session_state.page == 3:
    st.title("Skin Burn Classification - Interactive Model")
    st.markdown("Upload an image to detect the burn degree and get first aid guidance.")

    IMG_HEIGHT = 240
    IMG_WIDTH = 240
    CLASS_NAMES = ['No Skin burn', '1st degree', '2nd degree', '3rd degree']

    @st.cache_resource
    def load_model():
        interpreter = tf.lite.Interpreter(model_path="EfficientNetB3_skin_burn_model.tflite")
        interpreter.allocate_tensors()
        return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

    interpreter, input_details, output_details = load_model()

    instructions = {
        "No Skin burn": {"first_aid": {"general": "No visible burn detected. If you feel pain or symptoms, consult a doctor."}, "emergency_number": "123"},
        "1st degree": {"first_aid": {"do": ["Cool under running water (not ice-cold) for 10–15 minutes.", "Apply aloe vera or panthenol.", "Cover with sterile gauze.", "Take paracetamol if needed."], "avoid": ["Do not apply ice directly.", "Avoid toothpaste or butter.", "Do not pop blisters.", "Do not rub the area."]}, "emergency_number": "123"},
        "2nd degree": {"first_aid": {"do": ["Rinse under cool water for 15–30 minutes.", "Do not pop blisters.", "Apply antibiotic cream and cover with gauze.", "Drink water and seek medical advice."], "avoid": ["Do not use greasy substances.", "Avoid tight bandages.", "Do not leave open wounds uncovered."]}, "emergency_number": "123"},
        "3rd degree": {"first_aid": {"do": ["Call 123 immediately.", "Remove from burn source.", "Cover with clean cloth.", "Elevate burned area.", "Monitor breathing.", "Keep warm."], "avoid": ["Do not apply water or creams.", "Do not remove stuck clothing.", "Do not give food or drinks to unconscious person."]}, "emergency_number": "123"}
    }

    def preprocess_image(image: Image.Image):
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

    def predict(image):
        image_array = preprocess_image(image)
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_scale = output_details[0]['quantization'][0]
        output_zero_point = output_details[0]['quantization'][1]
        dequantized_output = (output_data.astype(np.float32) - output_zero_point) * output_scale
        predicted_idx = np.argmax(dequantized_output[0])
        predicted_class = CLASS_NAMES[predicted_idx]
        probabilities = dequantized_output[0].tolist()
        return predicted_class, probabilities

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Burn Degree"):
            predicted_class, probabilities = predict(image)
            st.subheader(f"Prediction: **{predicted_class}**")
            st.write("### Confidence Levels:")
            st.bar_chart({CLASS_NAMES[i]: probabilities[i] for i in range(len(CLASS_NAMES))})

            info = instructions[predicted_class]
            st.markdown("### First Aid Instructions:")
            if "general" in info["first_aid"]:
                st.info(info["first_aid"]["general"])
            else:
                st.success("**Do:**")
                for item in info["first_aid"]["do"]:
                    st.write(f"- {item}")
                st.error("**Avoid:**")
                for item in info["first_aid"]["avoid"]:
                    st.write(f"- {item}")
            st.warning(f"Emergency Number: {info['emergency_number']}")

    col1, col2 = st.columns(2)
    with col1:
        st.button("Left Arrow Back", on_click=prev_page, use_container_width=True)