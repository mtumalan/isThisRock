import streamlit as st
import librosa
import numpy as np
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