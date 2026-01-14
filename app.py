import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Skin Lesion Classifier Demo",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('skin_cancer_model_v1.keras')

with st.spinner('Loading model..'):
    model = load_model()

classes = {
    0: 'Actinic keratosis (Cancer - Pre)',
    1: 'Basal cell carcinoma (Cancer)',
    2: 'Benign keratosis (Benign)',
    3: 'Dermatofibroma (Benign)',
    4: 'Melanoma (Cancer)',
    5: 'Melanocytic nevi (Benign - Mole)',
    6: 'Vascular lesion (Benign)'
}

st.markdown("""
    <style>
        .stButton>button {width: 100%;}
        .stAlert {text-align: center;}
        div[data-testid="stImage"] {display: block; margin-left: auto; margin-right: auto;}
    </style>
""", unsafe_allow_html=True)

st.title("Skin Lesion Diagnosis")
st.markdown("---")

# upload section
st.subheader("1. Upload Image")
uploaded_file = st.file_uploader("Choose a dermoscopic image..", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    st.image(image, caption='Patient Image', use_container_width=True)

    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)

    # predict
    with st.spinner("Analyzing.."):
        predictions = model.predict(img_batch)

    score = np.max(predictions) * 100
    predicted_class = classes[np.argmax(predictions)]

    # results section
    st.markdown("---")
    st.subheader("2. Analysis Results")

    if "Cancer" in predicted_class :
        st.error(f"Detected **Diagnosis:** {predicted_class}") # to show cancer results in red
    else:
        st.success(f"Detected **Diagnosis:** {predicted_class}")    # to show benign results in green

    col_centered, _ = st.columns([1, 0.1])
    with col_centered:
        st.metric("Confidence Level", f"{score:.2f}%")

    # detail section
    st.markdown("---")
    st.subheader("3. Detailed Probabilities")

    probs = predictions[0]

    table_data = []
    for i in range(7):
        table_data.append({
            "Diagnosis": classes[i],
            "Probability": f"{probs[i] * 100:.2f}%",
            "Score": probs[i] * 100
        })

    df = pd.DataFrame(table_data)
    df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)

    # Show Table
    st.table(df[["Diagnosis", "Probability"]])

    # Show Chart
    st.bar_chart(df.set_index("Diagnosis")["Score"])