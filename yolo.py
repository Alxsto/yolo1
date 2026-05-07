import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# ---------------------------------------------------
# Seitenkonfiguration
# ---------------------------------------------------
st.set_page_config(
    page_title="YOLO Objekterkennung",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------
# Modernes Styling
# ---------------------------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }

        .title {
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            color: #1f2937;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 18px;
            margin-bottom: 30px;
        }

        .stButton>button {
            border-radius: 10px;
            padding: 0.5rem 1rem;
        }

        .result-box {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Titel
# ---------------------------------------------------
st.markdown('<div class="title">🤖 YOLO Objekterkennung</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Lade ein Bild hoch und erkenne automatisch Objekte mit YOLOv8</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# Modell laden
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

with st.spinner("Lade YOLO-Modell..."):
    model = load_model()

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.header("⚙️ Einstellungen")

confidence_threshold = st.sidebar.slider(
    "Minimale Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

st.sidebar.info(
    "Dieses Tool verwendet YOLOv8 zur Echtzeit-Objekterkennung."
)

# ---------------------------------------------------
# Upload-Bereich
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# Verarbeitung
# ---------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # YOLO Vorhersage
    with st.spinner("🔍 Analysiere Bild..."):
        results = model(img_array)

    result = results[0]

    # Bild mit Bounding Boxes
    annotated_frame = result.plot()
    annotated_frame = cv2.cvtColor(
        annotated_frame,
        cv2.COLOR_BGR2RGB
    )

    # Layout mit zwei Spalten
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📷 Originalbild")
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("### 🎯 Erkannte Objekte")
        st.image(annotated_frame, use_column_width=True)

    # ---------------------------------------------------
    # Ergebnisse sammeln
    # ---------------------------------------------------
    detections = []

    for box in result.boxes:
        conf = float(box.conf[0])

        if conf >= confidence_threshold:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            detections.append({
                "Objekt": label,
                "Confidence": f"{conf:.2%}"
            })

    # ---------------------------------------------------
    # Ergebnisse anzeigen
    # ---------------------------------------------------
    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )

    st.markdown("## 📋 Erkennungsergebnisse")

    if detections:
        df = pd.DataFrame(detections)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.success(f"✅ {len(detections)} Objekt(e) erkannt")
    else:
        st.warning("⚠️ Keine Objekte mit ausreichender Confidence erkannt")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")
st.caption("Erstellt mit Streamlit + YOLOv8")
