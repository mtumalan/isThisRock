import streamlit as st

st.title("Dataset: GTZAN Music Genre Classification")

# Introduction to the Dataset
st.header("Introduction to the GTZAN Dataset")
st.write("""
The **GTZAN Music Genre Dataset** is a well-known dataset for music genre classification. It contains audio files and features that allow for the classification of music into various genres. 
The dataset was introduced by George Tzanetakis in 2001 and has been widely used for genre recognition tasks in music information retrieval.

You can find the original dataset on Kaggle [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).
""")

# Using the 3-Second Version of the Dataset
st.header("Using the 3-Second Version of the Dataset")
st.write("""
While the original GTZAN dataset contains 30-second audio clips, for our project we are using a modified version of the dataset that splits each 30-second audio file into 3-second clips.

This version offers several benefits:

1. **Smaller Input Size**: By reducing the clip duration to 3 seconds, we decrease the size of each audio input, making the model training faster and more efficient.

2. **Increased Data Volume**: Each 30-second clip is split into 10 separate 3-second clips, increasing the number of training examples. This helps improve model generalization and robustness.

3. **Real-time Analysis**: The 3-second clips are more aligned with real-time genre classification, where short snippets of audio are classified on-the-fly.
""")

# Dataset Composition
st.header("Dataset Composition")
st.write("""
The modified dataset consists of multiple 3-second audio clips, with each original 30-second audio track split into 10 shorter clips. 
The genres in the dataset remain the same, including:

- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

This modification allows us to train the model more efficiently while maintaining the genre diversity.
""")

# Features in the Dataset
st.header("Features in the Dataset")
st.write("""
Each 3-second audio file is used to extract several important features, which we feed into the model for classification. These features include:

- **Mel-Frequency Cepstral Coefficients (MFCCs)**: Represent the power spectrum of the audio signal.
- **Chroma Features**: Capture harmonic and tonal characteristics.
- **Spectral Centroid and Bandwidth**: Measure the "brightness" and frequency distribution.
- **Root Mean Square (RMS)**: Measures the intensity of the signal.

These features help the model learn to distinguish between different music genres based on short snippets of sound.
""")