import streamlit as st
from PIL import Image
import tempfile
from predict import predict


st.title("Pneumonia Detection from Chest X-ray")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    prediction, confidence, heatmap_path = predict(image_path)

    st.subheader("Prediction")

    st.write("Result:", prediction)
    st.write("Confidence:", round(confidence*100,2), "%")

    st.subheader("GradCAM Explanation")

    heatmap = Image.open(heatmap_path)

    st.image(heatmap, caption="GradCAM Heatmap")