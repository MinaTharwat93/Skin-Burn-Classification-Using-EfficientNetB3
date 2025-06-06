# Skin Burn Classification using EfficientNetB3

This project centers around a deep learning model utilizing the EfficientNetB3 architecture to classify images of skin burns by degree (1st, 2nd, 3rd degree, or no burn). The development and training of this model are detailed in the `Skin Burn Classification Using EfficientNetB3.ipynb` Jupyter Notebook. For practical application, the trained model is exposed via a FastAPI application, which also provides relevant first aid information based on the classification.

## How it Works

This project leverages the `EfficientNetB3` deep learning model, developed and trained as detailed in the `Skin Burn Classification Using EfficientNetB3.ipynb` notebook and described in the "Model Development and Details" section above. The resulting TensorFlow Lite model (`EfficientNetB3_skin_burn_model.tflite`) is then utilized by a FastAPI application (`main.py`) to provide classification capabilities.

The process for classifying an uploaded image is as follows:

1.  **Image Reception:** The FastAPI application receives an uploaded image.
2.  **Preprocessing for Inference:** The image undergoes preprocessing steps (e.g., resizing to 240x240 pixels, RGB conversion, normalization, and quantization) via `main.py`. These steps are designed to match the input requirements of the pre-trained `EfficientNetB3` model, as established during its development in the Jupyter Notebook.
3.  **Model Inference:** The preprocessed image data is fed into the loaded `EfficientNetB3_skin_burn_model.tflite` model. The model performs inference and outputs predictions.
4.  **Classification and Output:** The model classifies the image into one of the predefined categories:
    *   No Skin burn
    *   1st degree
    *   2nd degree
    *   3rd degree
5.  **Response Generation:** The FastAPI application constructs a JSON response containing:
    *   The `predicted_class`.
    *   `probabilities` for each class.
    *   Relevant `first_aid` instructions for the predicted class.
    *   An `emergency_number` (currently a placeholder).

This flow allows the sophisticated image classification capabilities developed through deep learning research in the notebook to be accessed via a simple API interface.

## Model Development and Details

The core of this project is a deep learning model for skin burn classification, developed using Python and TensorFlow/Keras. The entire process, from data handling to model training and conversion, is detailed in the `Skin Burn Classification Using EfficientNetB3.ipynb` Jupyter Notebook.

### Data
The model was trained on a collection of images sourced from various public datasets, as listed in `Data.txt`. These images underwent preprocessing steps (e.g., resizing, augmentation if used) as defined in the notebook to prepare them for training.

### Architecture
The classification model is based on the **EfficientNetB3 architecture**. EfficientNets are a family of state-of-the-art convolutional neural networks (CNNs) known for achieving high accuracy with relatively fewer parameters (i.e., they are efficient). The B3 variant offers a good balance between performance and computational cost. The notebook details the specifics of how this architecture was implemented and potentially fine-tuned for the skin burn classification task.

### Training
The model was trained within the Jupyter Notebook using the prepared image dataset. This involved:
*   Splitting the data into training, validation, and test sets.
*   Defining an appropriate loss function for multi-class classification (e.g., categorical cross-entropy).
*   Choosing an optimizer (e.g., Adam).
*   Iteratively training the model over several epochs, monitoring performance on the validation set to prevent overfitting.

### Evaluation and Conversion
After training, the model's performance was evaluated using metrics such as accuracy, precision, recall, and F1-score on the test set. The specifics of this evaluation are available in the notebook.

For deployment, the trained TensorFlow/Keras model was converted into the **TensorFlow Lite (`.tflite`) format** (`EfficientNetB3_skin_burn_model.tflite`). This conversion optimizes the model for size and inference speed, making it suitable for use in applications, including mobile apps (like the intended Flutter integration) and the provided FastAPI.

---

## API Endpoints

### `/predict`

*   **Method:** `POST`
    *   _Note: While a `GET` endpoint also exists in the current `main.py` for `/predict`, using `POST` is recommended for file uploads._
*   **Description:** Uploads an image file for burn classification.
*   **Input:** Image file (e.g., JPEG, PNG). Formats supported: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.webp`.
*   **Request Body:** `multipart/form-data` with a `file` field containing the image.

*   **Success Response (200 OK):**
    *   **Content:** `application/json`
    *   **Example:**
        ```json
        {
            "predicted_class": "1st degree",
            "probabilities": {
                "No Skin burn": 0.05,
                "1st degree": 0.75,
                "2nd degree": 0.15,
                "3rd degree": 0.05
            },
            "first_aid": {
                "do": [
                    "Place the affected area under cool (not ice-cold) running water for 10-15 minutes.",
                    "Apply a soothing cream like aloe vera or panthenol.",
                    "Cover the burn with a sterile, loose gauze if needed.",
                    "Take over-the-counter pain relievers like paracetamol if there is pain."
                ],
                "avoid": [
                    "Do not apply ice directly to the burn.",
                    "Do not use toothpaste, butter, or home remedies.",
                    "Do not pop any blisters if they appear.",
                    "Do not rub the affected area."
                ]
            },
            "emergency_number": "123"
        }
        ```
*   **Error Responses:**
    *   `400 Bad Request`: If the file format is unsupported or the image is invalid/corrupted.
    *   `500 Internal Server Error`: If there's an issue during prediction.

## Setup and Running Locally

Follow these steps to set up and run the project on your local machine:

1.  **Prerequisites:**
    *   Python 3.7+

2.  **Clone the Repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_directory>
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    Make sure you have the `EfficientNetB3_skin_burn_model.tflite` file in the root directory of the project.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Application:**
    The application uses Uvicorn as the ASGI server.
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `--reload`: Enables auto-reloading when code changes, useful for development.
    *   The API will be accessible at `http://localhost:8000`. You can access the interactive API documentation (Swagger UI) at `http://localhost:8000/docs`.

## Data Sources

The image classification model was trained using data aggregated from various public datasets. The primary sources include:

*   [Skin Burns Dataset on Roboflow](https://universe.roboflow.com/aibuildersclub/skin-burns-4yoo2/dataset/3)
*   Kaggle Datasets:
    *   [Skin Burn Dataset by Mohammad Dimas Noufal](https://www.kaggle.com/datasets/mohammaddimasnoufal/skin-burn-dataset)
    *   [Skin Burn Dataset Cleaned v2 by Sayeem Zaman](https://www.kaggle.com/datasets/sayeemzzzaman/skin-burn-dataset-cleanedv2)
    *   [Last DataBurn by Fares Abbas AI](https://www.kaggle.com/datasets/faresabbasai2022/last-databurn)
    *   [Skin Burn Dataset by Fares Abbas AI (1)](https://www.kaggle.com/datasets/faresabbasai2022/skin-burn-dataset)
    *   [Skin Burns Dataset by Fares Abbas AI (2)](https://www.kaggle.com/datasets/faresabbasai2022/skin-burns-dataset)
    *   [Oily, Dry, and Normal Skin Types Dataset by Shakya Dissanayake](https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset) (It's noted in `Data.txt`, though its direct relevance to burn classification might be for non-burn images or a broader skin condition model).

Please refer to these sources for more information on the original datasets.

## First Aid Information

The API provides first aid suggestions based on the predicted burn classification. This information is intended for initial guidance only.

**Disclaimer:**
**The first aid information provided by this API is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read or received from this API.**

The emergency contact number provided in the API response is currently a placeholder (`123`) and should be updated to reflect appropriate local emergency services.

## Future Improvements

Some potential areas for future development and improvement include:

*   **Enhanced Model Accuracy:** Further training with a more diverse and larger dataset could improve classification accuracy and robustness.
*   **Localization:**
    *   Translate first aid instructions into multiple languages.
    *   Provide localized emergency contact numbers.
*   **Expanded Classification:** Include more granular classifications (e.g., distinguishing between superficial and deep 2nd-degree burns) or other skin conditions.
*   **User Feedback Mechanism:** Allow users to provide feedback on predictions to help identify areas for model improvement.
*   **Deployment Enhancements:**
    *   More robust error handling and logging.
    *   Secure the API if deployed in a public environment.
*   **Severity Assessment:** Beyond classification, potentially estimate the percentage of Total Body Surface Area (TBSA) affected, if feasible from an image.
*   **Integration with Telemedicine:** Explore possibilities for integrating with telemedicine platforms for quick consultation.
