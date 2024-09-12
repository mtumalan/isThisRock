import streamlit as st
import pandas as pd
import numpy as np
import librosa
import pyaudio
import io
import wave
import tempfile

def recordAudio(duration=5, fs=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=fs,
                    input=True,
                    frames_per_buffer=1024)

    frames = []
    st.write("Recording...")
    for _ in range(0, int(fs / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    st.write("Recording finished.")
    
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

def show_audio_extraction_page():
    st.title("Feature Extraction")

    # Introduction Section
    st.write("""
    We record a short audio clip and extract various audio features that are crucial in identifying the genre of the music.
    These features are used as inputs for our neural network model, helping it classify whether a given audio belongs to genres.
    """)

    # Libraries Section
    st.header("Libraries and Tools")
    st.write("""
    We use the following Python libraries for feature extraction and audio processing:
    - **librosa**: A Python package for music and audio analysis. It provides tools to extract essential features like Mel-frequency cepstral coefficients (MFCCs), chroma, and spectral centroids.
    - **pyaudio**: A Python library that provides Python bindings for PortAudio, enabling us to capture real-time audio recordings.
    """)

    # Feature Explanation Section
    st.header("Explanation of Extracted Features")
    st.write("""
    The features we extract from the audio signal are summarized below:
    
    1. **Chroma STFT**: This feature represents the energy of each pitch class (12-tone chromatic scale) in the audio. It's helpful in identifying harmonic and tonal characteristics.
    
    2. **Root Mean Square (RMS)**: Measures the power or intensity of the audio signal over time. Louder signals have higher RMS values.
    
    3. **Spectral Centroid**: Indicates where the "center of mass" of the spectrum is located. It's often related to the perceived brightness of a sound.
    
    4. **Spectral Bandwidth**: Describes the width of the spectrum, which can provide information about the range of frequencies in the audio.
    
    5. **MFCCs (Mel-Frequency Cepstral Coefficients)**: These are the most important features for audio classification. MFCCs represent the short-term power spectrum of the audio and are critical for distinguishing between different audio characteristics like timbre.
    """)

    # Recording Section
    st.header("Record and Extract Features")
    duration = st.slider("Select Recording Duration (seconds)", min_value=1, max_value=5, value=3)

    if st.button("Record"):
        # Record audio
        audio_data = recordAudio(duration)

        # Extract features
        features = extract_features(audio_data)
        st.write("Audio features extracted successfully!")

        # Display extracted features in a table
        st.write("Here are the extracted features:")
        st.table(pd.DataFrame(list(features.items()), columns=['Feature', 'Value']))

# To use the page in Streamlit
show_audio_extraction_page()