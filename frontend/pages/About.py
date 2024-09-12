import streamlit as st
from PIL import Image

# Page Title
st.title("About Is This Rock?")

# Introduction to the Project
st.header("Project Overview")
st.write("""
'Is This Rock?' is a project developed by our team to explore how neural networks can classify music genres based on live recordings. 
The goal was to create a model capable of identifying 10 different music genres by extracting features from audio signals in real-time, 
providing instant feedback on whether a given recording belongs to a particular genre.
""")

# Neural Network Information
st.header("Technology")
st.write("""
We built a neural network that extracts relevant features from live audio recordings, such as spectral properties, rhythm, and frequency content. 
The model was trained using a dataset containing labeled music genres, allowing it to distinguish between 10 different genres.
The core tools and technologies used in this project are:
- **Python** for the backend and data processing
- **TensorFlow** for neural network training
- **Streamlit** for the frontend interface
- **FastAPI** for API development and handling requests
""")

# Team Information
st.header("The Team")

# Load images using PIL
image_mt = Image.open("pages/static/mt.jpg")
image_fragoso = Image.open("pages/static/fragoso.jpg")
image_soto = Image.open("pages/static/soto.jpg")

# Create a centered layout with empty columns on the sides
col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])

with col2:
    st.image(image_mt, width=100)
    st.markdown("[Mauricio Tumalan](https://github.com/mtumalan)", unsafe_allow_html=True)
    st.write("Computer Science and Technology Student")

with col3:
    st.image(image_fragoso, width=100)
    st.markdown("[Carlos Fragoso](https://github.com/carlosfragoso21)", unsafe_allow_html=True)
    st.write("Computer Science and Technology Student")

with col4:
    st.image(image_soto, width=100)
    st.markdown("[Carlos Soto](https://github.com/CSA09)", unsafe_allow_html=True)
    st.write("Computer Science and Technology Student")

# Special Thanks Section
st.header("Special Thanks")

st.write("""
We would like to express our deepest gratitude to **Victor Manuel de la Cueva Hernández**, **Esteban Castillo Juarez** and **Gualberto Aguilar Torres** for their support and guidance throughout this project.
""")
