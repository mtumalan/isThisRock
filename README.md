# Is This Rock?

## Project Overview

This project aims to classify music genres in real-time using audio features extracted from short audio clips. We built a neural network model capable of recognizing 10 different music genres from live recordings, using a modified version of the **GTZAN Music Genre Dataset**. The model can determine whether a given audio belongs to genres like Rock, Jazz, Pop, and others.

## Features

- **Real-time Audio Classification**: Capture and classify live audio recordings.
- **Genre Prediction**: The model predicts one of 10 music genres based on extracted features.
- **Interactive Response**: If the genre predicted is "Rock," the interface provides a positive response, otherwise, a different message is shown depending on the predicted genre.

## Dataset

We used a modified version of the **GTZAN Music Genre Dataset**, where each 30-second track was split into 3-second clips. This modification increases the volume of training data and allows for efficient real-time classification. The dataset consists of the following 10 genres:
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

You can find the original dataset on [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

## How It Works

1. **Audio Recording**: The application uses **PyAudio** to capture live audio from your microphone.
2. **Feature Extraction**: We extract key audio features such as Mel-frequency cepstral coefficients (MFCCs), chroma features, spectral centroid, and others using the **Librosa** library.
3. **Neural Network Model**: A neural network built with **Keras** classifies the audio into one of the 10 genres.
4. **Real-time Prediction**: The system provides immediate feedback based on the predicted genre.

### Features Extracted:

- **MFCCs (Mel-frequency cepstral coefficients)**: Captures timbral features from the audio.
- **Chroma STFT**: Represents harmonic and tonal content.
- **Spectral Centroid**: Measures brightness.
- **Spectral Bandwidth**: Describes the width of the frequency range.
- **RMS (Root Mean Square)**: Measures the power or intensity of the audio.

## Model Architecture

The neural network is structured as follows:
- Input layer: Takes the extracted features.
- Two hidden layers using ReLU activation.
- Output layer with 10 neurons (one for each genre) using Softmax activation.

The model is trained using the **Adam** optimizer and **categorical cross-entropy** as the loss function. It was trained over 20 epochs using a 20% validation split.

## Getting Started

### Prerequisites

- **Python 3.11+**
- **TensorFlow** / **Keras**
- **Librosa**
- **PyAudio**
- **Streamlit**
- **Requests**

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/mtumalan/isThisRock/
    ```

2. Navigate to the project directory:
    ```bash
    cd isThisRock
    ```
    
3. Start the backend
      ```bash
      docker-compose up --build
      ```

4. Navigate to the frontend directory:
      ```bash
      cd frontend
      ```

5. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

6. Run the Streamlit application:
    ```bash
    streamlit run Extractor.py
    ```

## Usage

1. Start the application using Streamlit.
2. Adjust the recording duration via the slider.
3. Press "Record" to start recording live audio.
4. Once recorded, the audio is processed and classified into one of 10 genres.
5. If the predicted genre is "Rock," a positive message is displayed along with a relevant image. Otherwise, the predicted genre will be shown with an alternate response.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The **GTZAN Dataset** provided by George Tzanetakis.
- The **Librosa** and **PyAudio** libraries for feature extraction and audio capture.
- Our professors, Victor de la Cueva, Esteban Castillo and Gualberto Aguilar.

## Contributors

- Carlos Fragoso
- Mauricio Tumalan
