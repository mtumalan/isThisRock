import streamlit as st
from PIL import Image

st.title("Results: Model Performance")

# Description of the Metrics
st.header("Overview of Metrics")
st.write("""
The heatmap above represents the **Precision**, **Recall**, and **F1-Score** for each of the 10 music genres that our model was trained to classify.
Here’s a brief explanation of each metric:

- **Precision**: It measures the accuracy of the positive predictions. For example, the precision for the "Rock" genre indicates how many of the predicted "Rock" tracks were actually "Rock".
- **Recall**: It measures the ability of the model to capture all the positive examples. For example, the recall for "Jazz" indicates how well the model can identify all "Jazz" tracks.
- **F1-Score**: This is the harmonic mean of precision and recall, providing a balance between both. A higher F1-score means a better balance between precision and recall.
""")

# Display the image
image = Image.open("pages/static/table.jpg")  # Path to the image
st.image(image)