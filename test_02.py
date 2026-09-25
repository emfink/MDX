import streamlit as st
import numpy as np
import os
import joblib
import shutil
from PIL import Image
import time
from sklearn.ensemble import RandomForestClassifier

# --- Setup ---
DATA_DIR = "training_data"
MODEL_PATH = "simple_model.pkl"

st.set_page_config(page_title="DIY Teachable Machine", layout="wide")
os.makedirs(DATA_DIR, exist_ok=True)

# --- CSS to Tighten UI ---
st.markdown("""
    <style>
        .block-container { padding-top: 0.01rem; }
        h1 { font-size: 1.5rem !important; margin-bottom: 0.5rem; margin-top: -1rem;}
        h3 { font-size: 1.1rem !important; margin-bottom: 0.2rem; }
        .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-top: 10px; }
        [data-testid="stCameraInput"] { margin-bottom: -1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- Initialize Session State ---
if "model_ready" not in st.session_state:
    st.session_state.model_ready = os.path.exists(MODEL_PATH)
if "last_saved_name_a" not in st.session_state:
    st.session_state.last_saved_name_a = None
if "last_saved_name_b" not in st.session_state:
    st.session_state.last_saved_name_b = None

# --- Helper Functions ---
def save_image(class_name, img_file):
    clean_name = "".join([c for c in class_name if c.isalnum() or c in (' ', '_')]).strip()
    class_path = os.path.join(DATA_DIR, clean_name)
    os.makedirs(class_path, exist_ok=True)
    img = Image.open(img_file).convert("RGB")
    if img.height > img.width: img = img.rotate(90, expand=True)
    img.save(os.path.join(class_path, f"{int(time.time() * 1000)}.jpg"))

def preprocess_image(img_file):
    img = Image.open(img_file).resize((64, 64)).convert("L")
    return np.array(img).flatten()

def display_grid(class_path):
    if os.path.exists(class_path):
        imgs = sorted([f for f in os.listdir(class_path) if f.endswith(".jpg")])[::-1]
        if imgs:
            num_cols = 5
            for i in range(0, len(imgs), num_cols):
                cols = st.columns(num_cols)
                for j, img_name in enumerate(imgs[i : i + num_cols]):
                    with cols[j]: st.image(os.path.join(class_path, img_name), use_container_width=True)

# --- TOP BAR: RESET ---
st.title("🧠 DIY Teachable Machine")
if st.button("🗑️ Reset All Data", use_container_width=True, type="secondary"):
    if os.path.exists(DATA_DIR): shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
    st.session_state.model_ready = False
    # Use current camera names to prevent auto-save after reset
    st.session_state.last_saved_name_a = st.session_state.cam_a.name if st.session_state.cam_a else None
    st.session_state.last_saved_name_b = st.session_state.cam_b.name if st.session_state.cam_b else None
    st.rerun()

st.divider()

# --- MAIN COLUMNS ---
colA, colB, colPred = st.columns(3)

with colA:
    st.subheader("Object A")
    name_a = st.text_input("Label A", "Object A", key="txt_a", label_visibility="collapsed")
    cam_a = st.camera_input("Capture A", key="cam_a", label_visibility="collapsed")
    
    if cam_a and cam_a.name != st.session_state.last_saved_name_a:
        save_image(name_a, cam_a)
        st.session_state.last_saved_name_a = cam_a.name
        st.rerun()
    
    if st.button("Delete Last A", key="del_a", use_container_width=True):
        path = os.path.join(DATA_DIR, name_a)
        imgs = sorted([f for f in os.listdir(path) if f.endswith(".jpg")])
        if imgs: os.remove(os.path.join(path, imgs[-1]))
        st.rerun()
    display_grid(os.path.join(DATA_DIR, name_a))

with colB:
    st.subheader("Object B")
    name_b = st.text_input("Label B", "Object B", key="txt_b", label_visibility="collapsed")
    cam_b = st.camera_input("Capture B", key="cam_b", label_visibility="collapsed")
    
    if cam_b and cam_b.name != st.session_state.last_saved_name_b:
        save_image(name_b, cam_b)
        st.session_state.last_saved_name_b = cam_b.name
        st.rerun()
        
    if st.button("Delete Last B", key="del_b", use_container_width=True):
        path = os.path.join(DATA_DIR, name_b)
        imgs = sorted([f for f in os.listdir(path) if f.endswith(".jpg")])
        if imgs: os.remove(os.path.join(path, imgs[-1]))
        st.rerun()
    display_grid(os.path.join(DATA_DIR, name_b))

with colPred:
    st.subheader("Model & Prediction")
    if st.button("🚀 Train Model", use_container_width=True, type="primary"):
        classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        if len(classes) >= 2:
            X, y, valid = [], [], True
            for c in classes:
                files = [f for f in os.listdir(os.path.join(DATA_DIR, c)) if f.endswith(".jpg")]
                if len(files) < 3:
                    st.error(f"Need 3+ images for {c}")
                    valid = False; break
                for f in files:
                    X.append(preprocess_image(os.path.join(DATA_DIR, c, f)))
                    y.append(c)
            if valid:
                model = RandomForestClassifier(n_estimators=100)
                model.fit(X, y)
                joblib.dump(model, MODEL_PATH)
                st.session_state.model_ready = True
                st.rerun()

    if st.session_state.model_ready:
        live_img = st.camera_input("Predict", key="cam_pred", label_visibility="collapsed")
        
        # 🟢 Prediction Logic (No st.rerun here!)
        if live_img:
            model = joblib.load(MODEL_PATH)
            feat = preprocess_image(live_img)
            prediction = model.predict([feat])[0]
            confidence = np.max(model.predict_proba([feat])[0])
            
            # Display immediately under the camera
            st.metric("Result", prediction)
            st.progress(float(confidence))
            st.caption(f"Confidence: {confidence*100:.1f}%")
        else:
            st.info("Take a photo to predict.")
    else:
        st.warning("Train the model first.")
