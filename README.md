# Skin Lesion Classification Dashboard

This is a demo of a Streamlit dashboard for a Skin Lesion Classification project. The application utilizes a trained **Keras** model to predict skin lesion types based on input images.

## Available Classes
The model predicts the following 7 classes:

* **0:** Actinic keratosis *(Cancer - Pre)*
* **1:** Basal cell carcinoma *(Cancer)*
* **2:** Benign keratosis *(Benign)*
* **3:** Dermatofibroma *(Benign)*
* **4:** Melanoma *(Cancer)*
* **5:** Melanocytic nevi *(Benign - Mole)*
* **6:** Vascular lesion *(Benign)*

## Installation & Usage

1. **Install Dependencies**
   - Ensure you have the required packages installed:
   ```bash
   pip install -r requirements.txt
2. **Run the Application**
   - Launch the dashboard using Streamlit:
   ```bash
   streamlit run app.py