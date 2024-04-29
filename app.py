import streamlit as st
import numpy as np
import librosa
import tensorflow
from tensorflow.keras.models import Model

# Load the trained model
model = Model("model.h5")

# Define a function to predict the class of an input audio file
def predict_audio_class(audio_file):
    # Preprocess the new audio data
    librosa_audio_data, librosa_sample_rate = librosa.load(audio_file)
    mfccs_features = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=50)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    # Predict the class probabilities
    predicted_probabilities = model.predict(mfccs_scaled_features)

    # Find the class with the highest probability
    predicted_class = np.argmax(predicted_probabilities)

    # Map the predicted class index to the actual class label
    # You may need to define your class labels based on your dataset
    class_labels = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
                    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
    predicted_label = class_labels[predicted_class]

    return predicted_label

# Streamlit App
st.title("Audio Classification")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    # Display the audio file
    st.audio(uploaded_file, format='audio/wav')

    # Predict the class of the uploaded audio file
    predicted_class = predict_audio_class(uploaded_file)

    # Display the predicted class
    st.write("Predicted Class:", predicted_class)
