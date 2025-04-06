import streamlit as st
import librosa
import numpy as np
import joblib
import os

# App config
st.set_page_config(page_title="Grammar Score Predictor", page_icon="🎧")

# App title
st.title(" Grammar Score Prediction from Audio")
st.markdown("Upload a `.wav` file and get a predicted grammar score based on audio features (MFCC).")

# Load model
MODEL_FILENAME = "audio_score_model.pkl"
MODEL_PATH = os.path.join("..", "models", MODEL_FILENAME)

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        st.success(" Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        st.stop()
else:
    st.error(f" Model file not found at `{MODEL_PATH}`. Make sure it's saved with the correct name.")
    st.stop()

# Upload section
st.subheader(" Upload Audio File")
audio_file = st.file_uploader("Only .wav files are supported", type=["wav"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    try:
        with st.spinner(" Extracting features and making prediction..."):
            # Feature extraction
            y, sr = librosa.load(audio_file, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features = np.mean(mfcc, axis=1).reshape(1, -1)

            # Predict
            prediction = model.predict(features)[0]
            st.success(f" **Predicted Grammar Score:** {prediction:.2f}")

    except Exception as e:
        st.error(f"Error processing file: `{e}`")

# Footer
st.markdown("---")
st.caption("🔁 Built using Streamlit | 🎵 Features: Librosa MFCC | 🤖 ML Model: Scikit-learn")
