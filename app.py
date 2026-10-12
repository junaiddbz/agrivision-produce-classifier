"""
app.py
------
Streamlit interactive demo application for the Fruit & Vegetable Classifier.
Features 36-class produce recognition, preloaded samples for all 36 classes,
and native Grad-CAM visual explainability.

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub, then connect to https://streamlit.io/cloud
"""

from pathlib import Path
import streamlit as st
from PIL import Image

from src.gradcam import generate_gradcam
from src.predict import load_model, predict as run_predict

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriVision | Produce Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT = "results/best_model.pth"
TOP_K = 5
IMAGE_SIZE = 224
SAMPLES_DIR = Path("assets/samples")

# Class emoji dictionary
CLASS_EMOJI = {
    "apple": "🍎", "banana": "🍌", "beetroot": "🫚", "bell pepper": "🫑",
    "cabbage": "🥬", "capsicum": "🫑", "carrot": "🥕", "cauliflower": "🥦",
    "chilli pepper": "🌶️", "corn": "🌽", "cucumber": "🥒", "eggplant": "🍆",
    "garlic": "🧄", "ginger": "🫚", "grapes": "🍇", "jalepeno": "🌶️",
    "kiwi": "🥝", "lemon": "🍋", "lettuce": "🥬", "mango": "🥭",
    "onion": "🧅", "orange": "🍊", "paprika": "🌶️", "pear": "🍐",
    "peas": "🫛", "pineapple": "🍍", "pomegranate": "🍎", "potato": "🥔",
    "raddish": "🌱", "soy beans": "🫘", "spinach": "🥬", "sweetcorn": "🌽",
    "sweetpotato": "🍠", "tomato": "🍅", "turnip": "🌱", "watermelon": "🍉",
}

ALL_CLASSES = sorted(list(CLASS_EMOJI.keys()))


def get_emoji(label: str) -> str:
    for key, emoji in CLASS_EMOJI.items():
        if key in label.lower():
            return emoji
    return "🌿"


# ── Load model (cached across Streamlit reruns) ───────────────────────────────
@st.cache_resource(show_spinner="Loading trained model checkpoint...")
def get_model():
    if not Path(CHECKPOINT).exists():
        return None, None, None, None
    return load_model(CHECKPOINT)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("System Overview")
    st.markdown(
        """
        **Model Architecture:** EfficientNet-B0  
        **Parameters:** 4.05 Million  
        **Input Resolution:** 224 x 224 px  
        **Produce Classes:** 36 categories  
        **Test Accuracy:** 94.15%  
        **Macro F1-Score:** 93.96%  
        """
    )
    st.divider()

    st.subheader("Model Explainability")
    st.markdown(
        """
        Grad-CAM visualizes gradients flowing into the final convolutional stage (`features[8]`), highlighting key pixel features influencing decisions.
        """
    )
    st.divider()
    st.markdown("**GitHub Repository:** [junaiddbz/agrivision-produce-classifier](https://github.com/junaiddbz/agrivision-produce-classifier)")
    st.caption("Developed as an end-to-end Machine Learning portfolio project.")


# ── Main Header ───────────────────────────────────────────────────────────────
st.title("Fruit & Vegetable Classifier")
st.markdown(
    "Identify produce across **36 fine-grained categories** with high precision, "
    "explore prediction confidences, and inspect neural attention with **Grad-CAM explainability**."
)
st.divider()

# Model loading verification
model, class_to_idx, idx_to_class, device = get_model()

if model is None:
    st.error(
        f"Model checkpoint not found at `{CHECKPOINT}`. Please ensure training has completed."
    )
    st.stop()

# ── Input Selection: Sample produce or File Upload ─────────────────────────────
st.subheader("Choose Input Source")
input_mode = st.radio(
    "Select how you want to test the model:",
    options=[
        "Test Sample Produce (All 36 Classes)",
        "Upload Custom Image",
    ],
    horizontal=True,
    index=0,
)

input_image = None
image_caption = ""

if input_mode == "Test Sample Produce (All 36 Classes)":
    # Dropdown for all 36 classes
    col_sample, col_info = st.columns([2, 1])

    with col_sample:
        selected_class = st.selectbox(
            "Select any produce category to test:",
            options=ALL_CLASSES,
            format_func=lambda c: f"{get_emoji(c)} {c.title()}",
            index=1,  # Default to banana
            help="Select one of the 36 produce categories to load a verified test image.",
        )

    # Locate sample image in assets/samples or data/test
    sample_file = None
    if SAMPLES_DIR.exists():
        candidates = list(SAMPLES_DIR.glob(f"{selected_class}.*"))
        if candidates:
            sample_file = candidates[0]

    if sample_file is None:
        # Fallback to local data/test directory if available
        test_class_dir = Path("data/test") / selected_class
        if test_class_dir.exists():
            test_candidates = list(test_class_dir.glob("*.*"))
            if test_candidates:
                sample_file = test_candidates[0]

    if sample_file and sample_file.exists():
        input_image = Image.open(sample_file).convert("RGB")
        image_caption = f"Sample test image for '{selected_class.title()}'"
        with col_info:
            st.info(f"Loaded: **{get_emoji(selected_class)} {selected_class.title()}**")
    else:
        st.warning(f"No sample image found for '{selected_class}'. Please upload an image.")

else:
    uploaded_file = st.file_uploader(
        "Upload a produce image (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Select a photo containing a single fruit or vegetable.",
    )
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        image_caption = f"Uploaded photo: {uploaded_file.name}"

# ── Inference and Results Section ─────────────────────────────────────────────
if input_image is not None:
    st.divider()

    col_input, col_pred = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("Input Image")
        st.image(input_image, width="stretch", caption=image_caption)

    with col_pred:
        st.subheader("Predictions")
        with st.spinner("Classifying image..."):
            predictions = run_predict(
                input_image,
                model,
                idx_to_class,
                device,
                top_k=TOP_K,
                image_size=IMAGE_SIZE,
            )

        top_class, top_confidence = predictions[0]
        emoji_icon = get_emoji(top_class)

        st.metric(
            label="Top Prediction",
            value=f"{emoji_icon} {top_class.title()}",
            delta=f"{top_confidence * 100:.2f}% Confidence",
        )
        st.progress(top_confidence)

        st.markdown("**Top-5 Candidates:**")
        for rank, (cand_class, cand_conf) in enumerate(predictions, 1):
            cand_emoji = get_emoji(cand_class)
            st.markdown(
                f"**{rank}. {cand_emoji} {cand_class.title()}** — `{cand_conf * 100:.2f}%`"
            )
            st.progress(cand_conf)

    st.divider()

    # ── Grad-CAM Explainability Section ───────────────────────────────────────
    st.subheader("Grad-CAM Visual Explainability")
    st.markdown(
        "Inspect the internal attention of the convolutional neural network to understand "
        "which spatial visual cues (textures, edges, shape boundaries) drove this classification."
    )

    gradcam_enabled = st.toggle("Enable Grad-CAM Analysis", value=True)

    if gradcam_enabled:
        cam_col1, cam_col2, cam_col3 = st.columns([1.5, 1, 1])

        with cam_col1:
            target_class_choice = st.selectbox(
                "Target Class to Explain",
                options=[p[0] for p in predictions],
                format_func=lambda c: f"{get_emoji(c)} {c.title()} ({dict(predictions)[c]*100:.1f}%)",
                index=0,
                help="Select which class prediction to backpropagate gradients for.",
            )

        with cam_col2:
            colormap_choice = st.selectbox(
                "Heatmap Colormap",
                options=["JET", "VIRIDIS", "TURBO", "PLASMA", "INFERNO"],
                index=0,
                help="Select the false-color scale used to render activation intensity.",
            )

        with cam_col3:
            alpha_blend = st.slider(
                "Overlay Transparency",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
                help="Balance between raw image visibility and heatmap intensity.",
            )

        with st.spinner("Generating Grad-CAM activation maps..."):
            target_idx = class_to_idx[target_class_choice]
            orig_crop, overlay_img, filter_img, cam_array = generate_gradcam(
                image=input_image,
                model=model,
                target_class_idx=target_idx,
                device=device,
                image_size=IMAGE_SIZE,
                colormap_name=colormap_choice,
                alpha=alpha_blend,
            )

        st.markdown("#### View Options")
        view_mode = st.radio(
            "Select Display Mode:",
            options=[
                "Side-by-Side Comparison",
                "Grad-CAM Overlay",
                "Filter Only (Heatmap)",
                "Original Image (Model Crop)",
            ],
            horizontal=True,
            index=0,
        )

        if view_mode == "Side-by-Side Comparison":
            cam_view1, cam_view2, cam_view3 = st.columns(3)

            with cam_view1:
                st.markdown("**1. Original Image (Input Crop)**")
                st.image(orig_crop, width="stretch", caption="Normalized 224x224 crop fed to network")

            with cam_view2:
                st.markdown("**2. Filter Only (Activation Heatmap)**")
                st.image(filter_img, width="stretch", caption=f"Raw Grad-CAM saliency ({colormap_choice})")

            with cam_view3:
                st.markdown("**3. Grad-CAM Overlay**")
                st.image(overlay_img, width="stretch", caption=f"Weighted blend (alpha={alpha_blend})")

        elif view_mode == "Grad-CAM Overlay":
            disp_col, _ = st.columns([1.5, 1])
            with disp_col:
                st.image(
                    overlay_img,
                    width="stretch",
                    caption=f"Grad-CAM overlay for '{target_class_choice.title()}' (alpha={alpha_blend})",
                )

        elif view_mode == "Filter Only (Heatmap)":
            disp_col, _ = st.columns([1.5, 1])
            with disp_col:
                st.image(
                    filter_img,
                    width="stretch",
                    caption=f"Filter-only class activation map ({colormap_choice} scale) for '{target_class_choice.title()}'",
                )

        elif view_mode == "Original Image (Model Crop)":
            disp_col, _ = st.columns([1.5, 1])
            with disp_col:
                st.image(
                    orig_crop,
                    width="stretch",
                    caption="Preprocessed 224x224 center-crop evaluated by the model",
                )

        with st.expander("How to Interpret Grad-CAM Visualizations", expanded=False):
            st.markdown(
                """
                - **Warm Colors (Red, Yellow):** Indicate high neural activation. The model placed the heaviest weight on these image regions when predicting the selected class.
                - **Cool Colors (Blue, Violet):** Indicate low or zero activation. These regions did not meaningfully contribute to the decision.
                - **Filter Only:** Shows the raw spatial distribution of gradients backpropagated to the last convolutional block, free from image background artifacts.
                - **Overlay:** Fuses activations with the photograph to highlight specific visual structures (stems, skin texture, contours).
                """
            )

    st.caption("Powered by PyTorch EfficientNet-B0 and Gradient-Weighted Class Activation Mapping.")

else:
    st.info("Select a sample produce category or upload your own image to get started.")
