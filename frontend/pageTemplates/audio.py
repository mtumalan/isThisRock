import streamlit as st
import librosa
import numpy as np
import pyaudio
import io
import wave
import tempfile
import requests

# Record audio function
def recordAudio(duration=5, fs=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=fs,
                    input=True,
                    frames_per_buffer=1024)

    frames = []
    try:
        st.write("Recording...")
        for _ in range(0, int(fs / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)
        st.write("Recording finished.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    wav_fp = io.BytesIO()
    wf = wave.open(wav_fp, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    return wav_fp

# Extract features function
def extract_features(wav_fp):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(wav_fp.getvalue())
        temp.flush()

        # Load the audio file
        y, sr = librosa.load(temp.name, sr=None)

        # Extract audio features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        # Aggregate means and variances
        features_dict = {
            'chroma_stft_mean': np.mean(chroma_stft), 'chroma_stft_var': np.var(chroma_stft),
            'rms_mean': np.mean(rms), 'rms_var': np.var(rms),
            'spectral_centroid_mean': np.mean(spec_cent), 'spectral_centroid_var': np.var(spec_cent),
            'spectral_bandwidth_mean': np.mean(spec_bw), 'spectral_bandwidth_var': np.var(spec_bw)
        }

        # Add MFCCs
        for i in range(1, 21):
            features_dict[f'mfcc{i}_mean'] = np.mean(mfcc[i-1])
            features_dict[f'mfcc{i}_var'] = np.var(mfcc[i-1])

        return features_dict

# Send data to backend function
def send2backend(features):
    # Convert NumPy float32 to Python float
    features = {k: float(v) for k, v in features.items()}
    
    url = "http://localhost:8000/predict/"
    response = requests.post(url, json={"features": features})
    
    if response.status_code == 200:
        return response.json()  # Get the prediction result
    else:
        st.write("Error sending features to backend. Status code:", response.status_code)
        return None

# Main page function
def show_audio_page():
    st.title("Is this Rock?")
    st.write("This model predicts whether a song is Rock or not with audio features recorded live. Try it out!")

    duration = st.slider("Recording Duration (seconds)", 10, 30, 20)

    if st.button("Record"):
        audio_data = recordAudio(duration)

        # Extract features
        features = extract_features(audio_data)
        st.write("Features extracted successfully!")

        # Send features to backend and get prediction result
        prediction = send2backend(features)
        print(prediction)

        # Check the result and display the corresponding message and image
        if prediction:
            genre = prediction.get("prediction", "Unknown")
            if isinstance(genre, list):
                genre = genre[0]

            if genre == "rock":
                st.write("This is Rock! Awesome!")
                st.image("pageTemplates/static/tarazona.jpg", caption="Tarazona Approved")
            else:
                st.write(f"This is not Rock. It is {genre}. Maybe next time.")
        else:
            st.write("Failed to get a prediction.")