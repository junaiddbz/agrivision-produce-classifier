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

Design notes
------------
The look borrows from the produce aisle. The result is a colour-matched panel
with a PLU-style sticker for the confidence (bananas go yellow, tomatoes red,
spinach green), the model specs are laid out like a nutrition label, and a
thin crate-label stripe across the top of the page follows the same colour.
Recommended: copy `agrivision_config.toml` to `.streamlit/config.toml`.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
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

MODE_SAMPLE = "Try a sample"
MODE_UPLOAD = "Upload a photo"

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


def _lookup(table: dict, label: str, default):
    """Exact match first, then the longest key contained in the label.

    The longest-match rule matters: a plain substring scan finds "apple" inside
    "pineapple" and returns the wrong entry.
    """
    key = label.lower()
    if key in table:
        return table[key]
    for k in sorted(table, key=len, reverse=True):
        if k in key:
            return table[k]
    return default


def get_emoji(label: str) -> str:
    return _lookup(CLASS_EMOJI, label, "🌿")


# ── Colour families (each class takes the colour of the produce itself) ───────

@dataclass(frozen=True)
class Hue:
    accent: str  # sticker, top bar, stripe
    tint: str    # result panel background
    on: str      # text colour on top of the accent


RED = Hue("#C23A2E", "#F9E2DE", "#FFFFFF")
BEET = Hue("#9C2B55", "#F5DDE6", "#FFFFFF")
MELON = Hue("#D03F58", "#FADDE3", "#FFFFFF")
ORANGE = Hue("#C95F10", "#FBE7D4", "#FFFFFF")
YELLOW = Hue("#E4B10A", "#FBF1C6", "#2A2300")
GREEN = Hue("#2F7D3F", "#DDEFD9", "#FFFFFF")
LIME = Hue("#5F8F1F", "#E6F1CC", "#FFFFFF")
PURPLE = Hue("#6F3B8F", "#EAE0F2", "#FFFFFF")
TAN = Hue("#8A6540", "#F0E5D7", "#FFFFFF")
PALE = Hue("#667A60", "#E7EBE1", "#FFFFFF")

DEFAULT_HUE = GREEN

CLASS_HUE = {
    "apple": RED, "banana": YELLOW, "beetroot": BEET, "bell pepper": GREEN,
    "cabbage": LIME, "capsicum": GREEN, "carrot": ORANGE, "cauliflower": PALE,
    "chilli pepper": RED, "corn": YELLOW, "cucumber": GREEN, "eggplant": PURPLE,
    "garlic": PALE, "ginger": TAN, "grapes": PURPLE, "jalepeno": GREEN,
    "kiwi": LIME, "lemon": YELLOW, "lettuce": LIME, "mango": ORANGE,
    "onion": TAN, "orange": ORANGE, "paprika": RED, "pear": LIME,
    "peas": GREEN, "pineapple": YELLOW, "pomegranate": RED, "potato": TAN,
    "raddish": BEET, "soy beans": TAN, "spinach": GREEN, "sweetcorn": YELLOW,
    "sweetpotato": ORANGE, "tomato": RED, "turnip": PALE, "watermelon": MELON,
}


def get_hue(label: str) -> Hue:
    return _lookup(CLASS_HUE, label, DEFAULT_HUE)


# Approximate colour scales for the Grad-CAM legend (low to high influence)
CMAP_STOPS = {
    "JET": ["#00007f", "#0000ff", "#00ffff", "#ffff00", "#ff0000", "#7f0000"],
    "VIRIDIS": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    "TURBO": ["#30123b", "#4662d7", "#36aaf9", "#1ae4b6", "#72fe5e", "#c7ef34", "#faba39", "#f66b19", "#ca2a04", "#7a0403"],
    "PLASMA": ["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"],
    "INFERNO": ["#000004", "#420a68", "#932667", "#dd513a", "#fca50a", "#fcffa4"],
}


# ── Global CSS ────────────────────────────────────────────────────────────────

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800&display=swap');

:root {
    --paper: #F3F6EE;
    --panel: #FBFCF8;
    --ink: #12301F;
    --muted: #5A6B58;
    --line: #D5DDCB;
    --line-strong: #AEBBA0;
    --accent: #2F7D3F;
    --tint: #DDEFD9;
    --on: #FFFFFF;
    --font: 'Bricolage Grotesque', 'Segoe UI', system-ui, -apple-system, sans-serif;
    color-scheme: light;
}

/* ── Page surface and crate-label stripe */
.stApp {
    background: var(--paper);
    color: var(--ink);
    font-family: var(--font);
}
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 6px;
    background: var(--accent);
    z-index: 1000000;
}
.block-container { max-width: 1180px; padding-top: 2.6rem; padding-bottom: 4rem; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, .stDeployButton { visibility: hidden; }

/* ── Typography */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: var(--font);
    font-weight: 800;
    letter-spacing: -0.025em;
    color: var(--ink);
}
.stApp h3 { font-size: 1.9rem; line-height: 1.1; padding-top: 0.5rem; }
.stApp h4 { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.015em; }
.stApp p, .stApp li, .stApp label, .stApp td, .stApp th, .stApp .stMarkdown { font-family: var(--font); }
.stApp .stMarkdown p { line-height: 1.6; }
[data-testid="stCaptionContainer"] { color: var(--muted); }
.stApp code { background: rgba(18, 48, 31, 0.07); color: var(--ink); border-radius: 5px; padding: 1px 6px; font-size: 0.85em; }

/* ── Masthead */
.masthead { padding: 0 0 18px; }
.wordmark {
    font-weight: 800;
    font-size: clamp(3.2rem, 7vw, 5rem);
    line-height: 0.95;
    letter-spacing: -0.045em;
    color: var(--ink);
}
.mast-sub { max-width: 58ch; margin: 14px 0 0; color: var(--muted); font-size: 1.1rem; line-height: 1.55; }
.mast-sub strong { color: var(--ink); font-weight: 600; }

/* ── Sidebar and nutrition-label model facts */
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
.facts {
    background: #FFFFFF;
    border: 2px solid var(--ink);
    padding: 10px 14px 12px;
    color: var(--ink);
}
.facts-title { font-weight: 800; font-size: 2rem; line-height: 1; letter-spacing: -0.04em; }
.facts-serving { font-size: 0.85rem; margin-top: 4px; }
.facts-thick { height: 9px; background: var(--ink); margin: 8px 0 4px; }
.facts-big { display: flex; justify-content: space-between; align-items: baseline; font-weight: 800; }
.facts-big .label { font-size: 1.05rem; }
.facts-big .value { font-size: 2.1rem; letter-spacing: -0.03em; }
.facts-mid { height: 5px; background: var(--ink); margin: 4px 0 0; }
.facts-row { display: flex; justify-content: space-between; gap: 12px; padding: 6px 0; border-top: 1px solid var(--ink); font-size: 0.92rem; }
.facts-row:first-of-type { border-top: none; }
.facts-row b { font-weight: 700; text-align: right; }
.side-h { font-weight: 800; font-size: 1.15rem; letter-spacing: -0.02em; margin: 22px 0 4px; }
.side-note { font-size: 0.88rem; color: var(--muted); line-height: 1.55; }
.side-note a { color: var(--ink); font-weight: 600; }

/* ── Inputs */
[data-testid="stRadio"] [role="radiogroup"] { gap: 10px; }
[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: #FFFFFF;
    border: 1px solid var(--line-strong);
    border-radius: 999px;
    padding: 8px 18px 8px 12px;
    margin: 0;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background: var(--tint);
    border-color: var(--accent);
    box-shadow: inset 0 0 0 1px var(--accent);
}
[data-testid="stFileUploaderDropzone"] { background: var(--panel); border: 1.5px dashed var(--line-strong); border-radius: 18px; }
[data-baseweb="select"] > div { border-radius: 12px; border-color: var(--line-strong); background: #FFFFFF; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 16px; background: var(--panel); }
[data-testid="stAlert"] { border-radius: 14px; }
.stImage img { border-radius: 18px; }

/* ── Result panel */
.result {
    background: var(--tint);
    border: 1px solid var(--line);
    border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
    border-radius: 26px;
    padding: 28px 30px 26px;
}
.result-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; }
.result-label { color: var(--muted); font-size: 0.95rem; }
.result-name {
    font-weight: 800;
    font-size: clamp(2.6rem, 5vw, 4.3rem);
    line-height: 1;
    letter-spacing: -0.04em;
    margin-top: 6px;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.22em;
}
.result-emoji { font-size: 0.85em; }
.plu {
    flex: none;
    width: 118px;
    height: 90px;
    border-radius: 50%;
    background: var(--accent);
    color: var(--on);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transform: rotate(-7deg);
    box-shadow: inset 0 0 0 4px rgba(255, 255, 255, 0.55);
}
.plu-num { font-weight: 800; font-size: 1.6rem; line-height: 1; letter-spacing: -0.03em; }
.plu-cap { font-size: 0.74rem; font-weight: 600; margin-top: 3px; }
.result-margin { color: var(--muted); margin: 14px 0 22px; font-size: 1rem; line-height: 1.5; }
.result-margin strong { color: var(--ink); font-weight: 600; }

.cand { display: flex; flex-direction: column; gap: 11px; }
.cand-row { display: grid; grid-template-columns: 18px minmax(130px, 1.15fr) 2fr 54px; align-items: center; gap: 12px; font-size: 0.97rem; }
.cand-rank { color: var(--muted); font-variant-numeric: tabular-nums; }
.cand-name { color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.cand-track { display: block; height: 10px; background: rgba(18, 48, 31, 0.09); border-radius: 999px; overflow: hidden; }
.cand-bar { display: block; height: 100%; border-radius: 999px; background: rgba(18, 48, 31, 0.32); }
.cand-val { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
.cand-row.top .cand-bar { background: var(--accent); }
.cand-row.top .cand-name, .cand-row.top .cand-val { color: var(--ink); font-weight: 700; }

.hint {
    margin-top: 18px;
    padding: 12px 14px;
    background: rgba(255, 255, 255, 0.7);
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    font-size: 0.92rem;
    line-height: 1.5;
}

/* ── Grad-CAM figures and legend */
.fig-title { font-weight: 700; margin: 2px 0 8px; letter-spacing: -0.01em; }
.legend { max-width: 460px; margin: 18px 0 4px; }
.legend-bar { height: 10px; border-radius: 999px; border: 1px solid var(--line-strong); }
.legend-ends { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--muted); margin-top: 5px; }

/* ── Empty state and section rule */
.empty {
    border: 1.5px dashed var(--line-strong);
    border-radius: 22px;
    padding: 56px 28px;
    text-align: center;
    background: var(--panel);
    color: var(--muted);
    margin-top: 22px;
}
.empty-title { font-weight: 800; font-size: 1.9rem; letter-spacing: -0.03em; color: var(--ink); margin-bottom: 6px; }
.rule { height: 1px; background: var(--line); margin: 34px 0 6px; }
</style>
"""


def html(markup: str) -> None:
    """Render raw HTML/CSS, flattened so Markdown never mistakes indentation for code."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def inject_theme(hue: Hue) -> None:
    """Re-colour the stripe, sticker, bars and panel for the current produce."""
    html(f"<style>:root{{--accent:{hue.accent};--tint:{hue.tint};--on:{hue.on};}}</style>")


html(BASE_CSS)


# ── Load model (cached across Streamlit reruns) ───────────────────────────────
@st.cache_resource(show_spinner="Loading trained model checkpoint...")
def get_model():
    if not Path(CHECKPOINT).exists():
        return None, None, None, None
    return load_model(CHECKPOINT)


# ── Component builders ────────────────────────────────────────────────────────

def candidates_html(predictions: list[tuple[str, float]]) -> str:
    rows = []
    for rank, (cls, conf) in enumerate(predictions, 1):
        pct = conf * 100.0
        top = " top" if rank == 1 else ""
        rows.append(
            f'<div class="cand-row{top}">'
            f'<span class="cand-rank">{rank}</span>'
            f'<span class="cand-name">{get_emoji(cls)} {escape(cls.title())}</span>'
            f'<span class="cand-track"><span class="cand-bar" style="width:{max(pct, 0.5):.2f}%"></span></span>'
            f'<span class="cand-val">{pct:.1f}%</span></div>'
        )
    return f'<div class="cand">{"".join(rows)}</div>'


def result_panel_html(predictions: list[tuple[str, float]]) -> str:
    top_class, top_conf = predictions[0]
    pct = top_conf * 100.0

    if len(predictions) > 1:
        runner_class, runner_conf = predictions[1]
        margin = pct - runner_conf * 100.0
        margin_html = (
            f'<p class="result-margin"><strong>{margin:.1f} points</strong> ahead of '
            f"{get_emoji(runner_class)} {escape(runner_class.title())}, the next closest match.</p>"
        )
    else:
        margin_html = '<p class="result-margin"></p>'

    hint = ""
    if top_conf < 0.60:
        hint = (
            '<div class="hint"><strong>Low confidence.</strong> '
            "Try a photo of a single item, well lit, on a plain background.</div>"
        )

    return (
        '<section class="result">'
        '<div class="result-head"><div>'
        '<div class="result-label">Top prediction</div>'
        f'<div class="result-name"><span class="result-emoji">{get_emoji(top_class)}</span>{escape(top_class.title())}</div>'
        "</div>"
        f'<div class="plu"><span class="plu-num">{pct:.1f}%</span><span class="plu-cap">match</span></div>'
        "</div>"
        f"{margin_html}"
        f"{candidates_html(predictions)}"
        f"{hint}"
        "</section>"
    )


def legend_html(colormap: str) -> str:
    stops = ", ".join(CMAP_STOPS.get(colormap, CMAP_STOPS["JET"]))
    return (
        '<div class="legend">'
        f'<div class="legend-bar" style="background:linear-gradient(90deg,{stops})"></div>'
        '<div class="legend-ends"><span>Little influence</span><span>Strong influence</span></div>'
        "</div>"
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        html(
            """
            <div class="facts">
                <div class="facts-title">Model facts</div>
                <div class="facts-serving">Per image classified</div>
                <div class="facts-thick"></div>
                <div class="facts-big"><span class="label">Test accuracy</span><span class="value">94.15%</span></div>
                <div class="facts-mid"></div>
                <div class="facts-row"><span>Macro F1-score</span><b>93.96%</b></div>
                <div class="facts-row"><span>Architecture</span><b>EfficientNet-B0</b></div>
                <div class="facts-row"><span>Parameters</span><b>4.05 million</b></div>
                <div class="facts-row"><span>Input resolution</span><b>224 × 224 px</b></div>
                <div class="facts-row"><span>Produce classes</span><b>36</b></div>
            </div>
            """
        )
        html(
            """
            <div class="side-h">About the heatmap</div>
            <div class="side-note">
                Grad-CAM follows the gradients flowing into the final convolutional stage
                (<code>features[8]</code>) to highlight the pixels that pushed the model toward its answer.
            </div>
            <div class="side-h">Source code</div>
            <div class="side-note">
                <a href="https://github.com/junaiddbz/agrivision-produce-classifier">junaiddbz/agrivision-produce-classifier</a><br>
                Built as an end-to-end machine learning portfolio project.
            </div>
            """
        )


# ── Input selection ───────────────────────────────────────────────────────────

def render_input_section() -> tuple[Image.Image | None, str]:
    st.markdown("### Pick a photo")
    input_mode = st.radio(
        "How do you want to test the model?",
        options=[MODE_SAMPLE, MODE_UPLOAD],
        horizontal=True,
        index=0,
    )

    input_image: Image.Image | None = None
    image_caption = ""

    if input_mode == MODE_SAMPLE:
        col_sample, _ = st.columns([1.3, 1])
        with col_sample:
            selected_class = st.selectbox(
                "Choose a produce category",
                options=ALL_CLASSES,
                format_func=lambda c: f"{get_emoji(c)} {c.title()}",
                index=ALL_CLASSES.index("banana") if "banana" in ALL_CLASSES else 0,
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
            image_caption = f"Test image for {selected_class.title()}"
        else:
            st.warning(f"No sample image found for '{selected_class}'. Upload a photo instead.")
    else:
        uploaded_file = st.file_uploader(
            "Upload a produce image (JPG, PNG or WebP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Select a photo containing a single fruit or vegetable.",
        )
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file).convert("RGB")
            image_caption = f"Uploaded photo: {uploaded_file.name}"

    return input_image, image_caption


# ── Grad-CAM section ──────────────────────────────────────────────────────────

def render_gradcam_section(input_image, model, class_to_idx, device, predictions) -> None:
    html('<div class="rule"></div>')
    st.markdown("### What the model looked at")
    st.markdown(
        "Grad-CAM shows which parts of the photo, such as skin texture, stems and outlines, "
        "pushed the network toward a class."
    )

    if not st.toggle("Show Grad-CAM", value=True):
        return

    cam_col1, cam_col2, cam_col3 = st.columns([1.5, 1, 1], gap="large")

    with cam_col1:
        target_class_choice = st.selectbox(
            "Class to explain",
            options=[p[0] for p in predictions],
            format_func=lambda c: f"{get_emoji(c)} {c.title()} ({dict(predictions)[c] * 100:.1f}%)",
            index=0,
            help="Select which class prediction to backpropagate gradients for.",
        )
    with cam_col2:
        colormap_choice = st.selectbox(
            "Colormap",
            options=["JET", "VIRIDIS", "TURBO", "PLASMA", "INFERNO"],
            index=0,
            help="Select the false-color scale used to render activation intensity.",
        )
    with cam_col3:
        alpha_blend = st.slider(
            "Overlay opacity",
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

    view_mode = st.radio(
        "View",
        options=[
            "Side by side",
            "Overlay",
            "Heatmap only",
            "Model input",
        ],
        horizontal=True,
        index=0,
    )

    if view_mode == "Side by side":
        v1, v2, v3 = st.columns(3, gap="medium")
        with v1:
            html('<div class="fig-title">Model input</div>')
            st.image(orig_crop, width="stretch", caption="The 224×224 crop the network sees")
        with v2:
            html('<div class="fig-title">Heatmap</div>')
            st.image(filter_img, width="stretch", caption=f"Raw Grad-CAM saliency ({colormap_choice})")
        with v3:
            html('<div class="fig-title">Overlay</div>')
            st.image(overlay_img, width="stretch", caption=f"Heatmap blended at {alpha_blend:.2f} opacity")
    else:
        disp_col, _ = st.columns([1.5, 1])
        with disp_col:
            if view_mode == "Overlay":
                st.image(
                    overlay_img,
                    width="stretch",
                    caption=f"Grad-CAM overlay for '{target_class_choice.title()}' ({alpha_blend:.2f} opacity)",
                )
            elif view_mode == "Heatmap only":
                st.image(
                    filter_img,
                    width="stretch",
                    caption=f"Class activation map ({colormap_choice}) for '{target_class_choice.title()}'",
                )
            else:
                st.image(
                    orig_crop,
                    width="stretch",
                    caption="Preprocessed 224×224 center crop evaluated by the model",
                )

    if view_mode != "Model input":
        html(legend_html(colormap_choice))

    with st.expander("How to read these maps", expanded=False):
        st.markdown(
            """
            - **Strong influence:** the model leaned hardest on these regions when predicting the selected class. The legend above shows which colours mean this for the scale you picked.
            - **Little influence:** these regions did not meaningfully contribute to the decision.
            - **Heatmap only:** the raw spatial distribution of gradients reaching the last convolutional block, with no photo behind it.
            - **Overlay:** the heatmap fused with the photograph, so you can see which structures (stems, skin texture, contours) carry the signal.
            """
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    render_sidebar()

    html(
        """
        <header class="masthead">
            <div class="wordmark">AgriVision</div>
            <p class="mast-sub">
                Identify <strong>36 fruits and vegetables</strong> from a photo, see how sure the model
                is, and check what it was looking at.
            </p>
        </header>
        """
    )

    # Model loading verification
    model, class_to_idx, idx_to_class, device = get_model()
    if model is None:
        st.error(f"Model checkpoint not found at `{CHECKPOINT}`. Make sure training has completed.")
        st.stop()

    input_image, image_caption = render_input_section()

    if input_image is None:
        inject_theme(DEFAULT_HUE)
        html(
            """
            <div class="empty">
                <div class="empty-title">Nothing to classify yet</div>
                Pick a sample category or upload your own photo above to get started.
            </div>
            """
        )
        return

    # ── Inference
    with st.spinner("Classifying image..."):
        predictions = run_predict(
            input_image,
            model,
            idx_to_class,
            device,
            top_k=TOP_K,
            image_size=IMAGE_SIZE,
        )

    top_class, _ = predictions[0]
    inject_theme(get_hue(top_class))

    st.markdown("&nbsp;", unsafe_allow_html=True)
    col_input, col_pred = st.columns([1, 1.05], gap="large")
    with col_input:
        st.image(input_image, width="stretch", caption=image_caption)
    with col_pred:
        html(result_panel_html(predictions))

    render_gradcam_section(input_image, model, class_to_idx, device, predictions)

    st.caption("Powered by PyTorch EfficientNet-B0 and Gradient-weighted Class Activation Mapping.")


main()
