import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time

# =========================
# STREAMLIT PAGE CONFIG
# =========================
st.set_page_config(page_title="🛡️ Live Liveness Detection", layout="wide")
st.title("🛡️ Live Liveness Detection")

# Initialize session state for the webcam running status
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False

# =========================
# LOAD KERAS MODELS
# =========================
@st.cache_resource
def load_keras_models():
    try:
        model1 = tf.keras.models.load_model("model.h5")
        model2 = tf.keras.models.load_model("model_1.h5")
        return model1, model2, True
    except Exception as e:
        # Return False status if loading fails
        st.error(f"❌ Error loading Keras models: {e}. Please ensure 'model.h5' and 'model_1.h5' exist.")
        return None, None, False

model1, model2, models_loaded = load_keras_models()

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_keras(frame):
    # Resize to 150x150, normalize, and add batch dimension
    frame = cv2.resize(frame, (150, 150))
    frame = frame.astype("float32") / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# =========================
# PREDICTION FUNCTION
# =========================
def predict_keras(frame):
    if not models_loaded:
        return "MODEL ERROR", 0.0

    k_frame = preprocess_keras(frame)

    pred1 = model1.predict(k_frame, verbose=0)[0][0]
    pred2 = model2.predict(k_frame, verbose=0)[0][0]
    final_score = (pred1 + pred2) / 2.0

    label = "REAL" if final_score >= 0.5 else "SPOOF"
    return label, final_score

# =========================
# CORE WEBCAM LOOP FUNCTION
# =========================
def run_webcam_detection(stframe_placeholder):
    # This function executes the continuous video loop
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        stframe_placeholder.error("Cannot access camera. Please check permissions.")
        st.session_state.webcam_running = False
        return # Exit if camera fails to open

    st.info("Live Liveness Check Running...")

    # Loop needs to check the session state flag in every iteration
    while st.session_state.webcam_running:
        ret, frame = cap.read()

        # Check if the "Stop" button was pressed since the last iteration
        # This is the key to responsive stopping in Streamlit
        if not st.session_state.webcam_running:
            break

        if not ret:
            stframe_placeholder.error("Camera stream ended unexpectedly.")
            st.session_state.webcam_running = False
            break

        frame = cv2.flip(frame, 1)
        label, score = predict_keras(frame)

        # Draw detection box and text
        color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
        text = f"{label} ({score:.2f})"

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (w - 10, h - 10), color, 5)

        cv2.putText(frame, text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        stframe_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Reduced sleep time for better responsiveness
        time.sleep(0.001)

        # --- CLEANUP ---
    if cap and cap.isOpened():
        cap.release()
    st.session_state.webcam_running = False
    stframe_placeholder.empty()
    st.warning("Webcam feed stopped. Press 'Start' to resume.")
    # Use rerun to ensure button states update correctly
    st.rerun()

# =========================
# SIDEBAR UI AND CONTROLS
# =========================
st.sidebar.title("⚙️ Controls")

# Create Start/Stop buttons
col1, col2 = st.sidebar.columns(2)

if col1.button("▶️ Start Liveness Check", use_container_width=True, disabled=st.session_state.webcam_running or not models_loaded):
    # Set running state to True and rerun to start the loop
    st.session_state.webcam_running = True
    st.rerun()

if col2.button("⏹️ Stop Webcam", use_container_width=True, disabled=not st.session_state.webcam_running):
    # Set running state to False to break the loop
    st.session_state.webcam_running = False
    # NO RERUN HERE. The loop needs to check the state and exit naturally first.
    # The run_webcam_detection function will call st.rerun() after cleanup.

# =========================
# WEBCAM FEED LOGIC (Main Entry Point)
# =========================
stframe = st.empty() # Placeholder for the video feed

if st.session_state.webcam_running and models_loaded:
    # If the state is running, enter the video loop function
    run_webcam_detection(stframe)
elif not models_loaded:
    stframe.warning("Liveness detection cannot start. Please resolve model loading errors.")
else:
    # Initial or stopped state message
    stframe.info("Press 'Start Liveness Check' in the sidebar to begin.")