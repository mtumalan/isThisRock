import streamlit as st
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# Page Title
st.title("Understanding the Neural Network")

# Introduction Section
st.header("How the Neural Network Works")
st.write("""
Our neural network is designed to classify music genres based on audio features extracted from 3-second audio clips. 
The neural network follows a series of steps that include data preprocessing, feature scaling, model definition, training, and prediction.
""")

# Data Preprocessing Section
st.header("Data Preprocessing")
st.write("""
The first step in building the neural network is to load the dataset, remove unnecessary columns, and separate the features (X) from the labels (Y). 
We use the following steps to preprocess the data:
1. **Remove unwanted columns**: These include filename, length, and specific columns that do not contribute to the model.
2. **Label encoding**: We encode the genre labels as integers so that the neural network can process them.
3. **Feature scaling**: We use `StandardScaler` to normalize the feature data, ensuring that all features contribute equally to the model training.
""")

csv_file = 'pages/static/trainData.csv'

# Load and display sample data

def cargar_datos():
    df = pd.read_csv(csv_file)
    return df

data = cargar_datos()
st.write("Here is a preview of our dataset after preprocessing:")
st.dataframe(data)

# Model Architecture Section
st.header("Neural Network Architecture")
st.write("""
Our neural network is a fully connected feedforward network, structured as follows:
- **Input Layer**: The number of neurons corresponds to the number of features in the dataset.
- **Hidden Layers**: We use two hidden layers. The first hidden layer contains neurons equal to the number of features, and the second contains half that number. Each layer uses the ReLU activation function.
- **Output Layer**: The output layer has one neuron for each genre we are trying to classify, with a softmax activation function for multi-class classification.
""")

# Display the model architecture
st.code("""
model = Sequential()
model.add(Dense(num_features, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(int(num_features / 2), activation='relu'))
model.add(Dense(len(np.unique(trainLabels)), activation='softmax'))
""", language='python')

# Model Training Section
st.header("Model Training")
st.write("""
We compile the model using the Adam optimizer and sparse categorical cross-entropy as the loss function. The model is trained for 20 epochs with a batch size of 200 and a validation split of 20%.
""")

st.code("""
model.compile(optimizer=Adam(learning_rate=0.005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=200, validation_split=0.2)
""", language='python')

# Model Evaluation Section
st.header("Model Evaluation")
st.write("""
After training, we evaluate the model using the test dataset to measure its accuracy.
""")

st.code("""
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
""", language='python')

# Closing Section
st.header("Prediction")
st.write("""
To make predictions with the model, we process new audio data in the same way as the training data, normalize it, and feed it into the model for genre classification.
""")

st.code("""
X_test = scaler.transform(new_data)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
""", language='python')
