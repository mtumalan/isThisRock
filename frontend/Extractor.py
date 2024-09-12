import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="Is this Rock?",
    page_icon="🎸",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'audio'

# Control the flow between pages using explicit function calls
if st.session_state.page == 'audio':
    from pageTemplates.audio import show_audio_page
    show_audio_page()