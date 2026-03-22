import streamlit as st
from PIL import Image
import tempfile
import torch
from predict import predict, predict_raw
from lung_segmentation import segment_lung
from gradcam import generate_gradcam, generate_gradcam_raw


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/efficientnet_best.pth"


st.title("Pneumonia Detection from Chest X-ray")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    # --- Step 1: Show Lung Segmentation ---
    st.subheader("Step 1 — Lung Segmentation")
    st.caption("The system isolates the lung regions before classification, ensuring the model focuses only on relevant anatomy.")

    original, mask, segmented = segment_lung(image_path)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original X-ray", use_container_width=True)
    with col2:
        st.image(segmented, caption="Segmented Lungs", use_container_width=True)

    st.divider()

    # --- Step 2: Side-by-Side Comparison ---
    st.subheader("Step 2 — Comparison: With vs Without Segmentation")
    st.caption("The same model processes both the raw X-ray and the segmented version. Compare the predictions and Grad-CAM heatmaps to see how segmentation influences model behavior.")

    # Run both pipelines
    with st.spinner("Running predictions..."):
        # Without segmentation (raw image)
        raw_prediction, raw_confidence = predict_raw(image_path)
        raw_heatmap_path = generate_gradcam_raw(image_path, model_path, device)

        # With segmentation
        seg_prediction, seg_confidence, seg_heatmap_path = predict(image_path)

    col_raw, col_seg = st.columns(2)

    with col_raw:
        st.markdown("### 🔴 Without Segmentation")
        st.metric(label="Prediction", value=raw_prediction)
        st.metric(label="Confidence", value=f"{round(raw_confidence*100, 2)}%")
        st.image(
            Image.open(raw_heatmap_path),
            caption="Grad-CAM — Full Image (may focus on non-lung areas)",
            use_container_width=True
        )

    with col_seg:
        st.markdown("### 🟢 With Segmentation")
        st.metric(label="Prediction", value=seg_prediction)
        st.metric(label="Confidence", value=f"{round(seg_confidence*100, 2)}%")
        st.image(
            Image.open(seg_heatmap_path),
            caption="Grad-CAM — Segmented Lungs (focused on lung tissue)",
            use_container_width=True
        )

    st.divider()

    # --- Step 3: Verdict ---
    st.subheader("Step 3 — Analysis Verdict")

    # Compare results
    same_prediction = raw_prediction == seg_prediction
    confidence_diff = abs(seg_confidence - raw_confidence) * 100

    if same_prediction:
        st.success(
            f"✅ **Both pipelines agree:** {seg_prediction}\n\n"
            f"- Without Segmentation: **{round(raw_confidence*100, 2)}%** confidence\n"
            f"- With Segmentation: **{round(seg_confidence*100, 2)}%** confidence\n\n"
            f"Confidence difference: **{round(confidence_diff, 2)}%**"
        )
    else:
        st.warning(
            f"⚠️ **Pipelines disagree!**\n\n"
            f"- Without Segmentation: **{raw_prediction}** ({round(raw_confidence*100, 2)}%)\n"
            f"- With Segmentation: **{seg_prediction}** ({round(seg_confidence*100, 2)}%)\n\n"
            f"This demonstrates that noise from non-lung regions can mislead the model. "
            f"The segmented result is more reliable as it focuses exclusively on lung tissue."
        )

    st.info(
        "💡 **Why segmentation matters:** Lung segmentation removes irrelevant structures "
        "(ribs, diaphragm, cardiac silhouette) that can confuse the model. "
        "Compare the Grad-CAM heatmaps above — the segmented version focuses its attention "
        "on clinically relevant lung regions, leading to more trustworthy predictions."
    )