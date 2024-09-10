import streamlit as st
import librosa
import numpy as np
import pyaudio
import io
import wave
import soundfile as sf
import tempfile

def recordAudio(duration = 5, fs = 44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, 
                    channels=1, 
                    rate=fs, 
                    input=True, 
                    input_device_index=16,
                    frames_per_buffer=1024)
    
    st.write("Recording...")
    frames = []
    
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
    # Save into temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(wav_fp.getvalue())
        temp.flush()

        # Load the audio file
        y, sr = librosa.load(temp.name, sr=None)
        print(f"Audio signal shape: {y.shape}")

        # Extract audio features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # Print feature shapes to debug
        print(f"Chroma STFT shape: {chroma_stft.shape}")
        print(f"RMS shape: {rms.shape}")
        print(f"Spectral Centroid shape: {spec_cent.shape}")
        print(f"Spectral Bandwidth shape: {spec_bw.shape}")
        print(f"MFCC shape: {mfcc.shape}")
        
        # Aggregate features (mean across frames)
        chroma_stft_mean = np.mean(chroma_stft, axis=1)
        rms_mean = np.mean(rms, axis=1)
        spec_cent_mean = np.mean(spec_cent, axis=1)
        spec_bw_mean = np.mean(spec_bw, axis=1)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Concatenate all features
        features = np.concatenate([
            chroma_stft_mean,
            rms_mean,
            spec_cent_mean,
            spec_bw_mean,
            mfcc_mean
        ])
        
        print(f"Features shape: {features.shape}")
        
        return features

def show_audio_page():
    st.title("Is this Rock?")
    st.write("This model predicts whether a song is Rock or not with audio features recorded live. Try it out!")

    duration = st.slider("Recording Duration (seconds)", 1, 10, 5)

    if st.button("Record"):
        audio_data = recordAudio(duration)

        # Extract features
        features = extract_features(audio_data)
        st.write("Features extracted successfully!")
        st.write(features)