import streamlit as st
import numpy as np
import os
import joblib
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import shutil

# --- Setup Directories and Initial Clearing ---
DATA_DIR = "training_data"
MODEL_PATH = "simple_model.pkl"

# CRITICAL: Erase pictures from previous runs on startup
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        st.toast("Wiped training data from previous run.", icon="🗑️")
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    os.makedirs(DATA_DIR)

st.set_page_config(page_title="Stacked Teachable Machine", layout="wide")
st.title("🤖 Stacked Teachable Machine")
st.caption("Captures are reset on every refresh. Stacking inputs for clarity.")

# Function to capture images for a given class
def capture_section(class_id, default_name):
    st.subheader(f"Capture {default_name}")
    class_name_input = st.text_input(f"Name for {class_id}", default_name, key=f"{class_id}_name").strip()
    class_path = os.path.join(DATA_DIR, class_name_input)
    if not os.path.exists(class_path): 
        os.makedirs(class_path)

    col_cam, col_info = st.columns([2, 1])
    
    with col_cam:
        img_file = st.camera_input(f"Snapshot for {class_name_input}", key=f"{class_id}_cam")
    
    with col_info:
        if img_file:
            img = Image.open(img_file)
            if st.button(f"✅ Add to {class_name_input}", key=f"{class_id}_add"):
                img_count = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
                img.save(os.path.join(class_path, f"{img_count}.jpg"))
                st.toast(f"Saved to {class_name_input}!")

        current_count = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
        st.write(f"**Total images for {class_name_input}:** {current_count}")
        st.divider()
    
    return class_name_input

# --- Main Interface with Stacking ---
col1, col2 = st.columns([2, 1])

with col1:
    # Stacking A and B on top of each other
    name_a = capture_section("A", "Object A")
    st.markdown("---")
    name_b = capture_section("B", "Object B")

with col2:
    st.subheader("⚙️ Training & Prediction")
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    if st.button("🚀 Train Model", use_container_width=True):
        X, y = [], []
        
        # Ensure only folders with images are processed
        valid_classes = [c for c in classes if len([f for f in os.listdir(os.path.join(DATA_DIR, c)) if f.endswith('.jpg')]) > 0]

        if len(valid_classes) < 2:
            st.error("Add at least 1 image to both Object A and Object B to train!")
        else:
            with st.spinner("Training..."):
                for label in valid_classes:
                    path = os.path.join(DATA_DIR, label)
                    for img_name in os.listdir(path):
                        if img_name.endswith(".jpg"):
                            # Resize and convert for stable training
                            img_train = Image.open(os.path.join(path, img_name)).resize((64, 64)).convert('L')
                            X.append(np.array(img_train).flatten())
                            y.append(label)
                
                model = KNeighborsClassifier(n_neighbors=3)
                model.fit(X, y)
                joblib.dump(model, MODEL_PATH)
                st.success("Model Trained!")

    st.divider()
    
    # Prediction Section (Using the Camera Inputs)
    st.subheader("🔍 Real-time Prediction")
    
    # We check if a model exists before running prediction logic
    if os.path.exists(MODEL_PATH):
        # We need a function to process the prediction for cleaner code
        def predict_img(img_handle, model_path_str):
            if img_handle:
                model = joblib.load(model_path_str)
                test_img = np.array(Image.open(img_handle).resize((64, 64)).convert('L')).flatten()
                prediction = model.predict([test_img])[0]
                return prediction
            return None

        # Predict based on the active camera stream (whichever one was last 'snapped')
        pred_a = predict_img(st.session_state.get("A_cam"), MODEL_PATH)
        pred_b = predict_img(st.session_state.get("B_cam"), MODEL_PATH)
        
        # Displaying the most relevant prediction
        if pred_a:
            st.markdown(f"### Result (from A's Camera): **{pred_a}**")
        elif pred_b:
            st.markdown(f"### Result (from B's Camera): **{pred_b}**")
        else:
            st.info("Take a photo with either camera to test the model.")
            
    else:
        st.info("Train the model first to see predictions.")

# --- Photo Gallery Section ---
st.divider()
st.subheader("🖼️ Your Dataset Gallery")

current_classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

if current_classes:
    # Create a tab for each object class
    tabs = st.tabs(current_classes)
    for i, tab in enumerate(tabs):
        with tab:
            current_class_name = current_classes[i]
            p = os.path.join(DATA_DIR, current_class_name)
            images = sorted([f for f in os.listdir(p) if f.endswith('.jpg')])
            
            if not images:
                st.info("No images yet. Take some photos!")
            else:
                # Show the last 6 images taken in a grid
                num_to_show = 6
                cols = st.columns(num_to_show)
                display_images = list(reversed(images))[:num_to_show]
                
                for idx, img_name in enumerate(display_images):
                    with cols[idx % num_to_show]:
                        display_img = Image.open(os.path.join(p, img_name))
                        st.image(display_img, use_container_width=True, caption=f"Sample {idx+1}")