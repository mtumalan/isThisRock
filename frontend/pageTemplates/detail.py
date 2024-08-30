import streamlit as st
import requests

def show_details_page():
    # Ensure the selected track is in the session state
    if 'selected_track' not in st.session_state:
        st.write("No track selected. Please go back and select a track.")
        if st.button("Back to Search"):
            st.session_state.page = 'search'
            st.experimental_rerun()
    else:
        track = st.session_state.selected_track
        st.title(f"Details for {track['name']} by {', '.join(track['artists'])}")

        # Fetch detailed song features from the backend API
        response = requests.get(f"http://api:8000/features", params={"track_id": track['id']})
        features = response.json()

        if features:
            st.write("Song Features:")
            st.write(f" - Danceability: {features['danceability']}")
            st.write(f" - Energy: {features['energy']}")
            st.write(f" - Key: {features['key']}")
            st.write(f" - Loudness: {features['loudness']}")
            st.write(f" - Mode: {features['mode']}")
            st.write(f" - Speechiness: {features['speechiness']}")
            st.write(f" - Acousticness: {features['acousticness']}")
            st.write(f" - Instrumentalness: {features['instrumentalness']}")
            st.write(f" - Liveness: {features['liveness']}")
            st.write(f" - Valence: {features['valence']}")
            st.write(f" - Tempo: {features['tempo']}")
            st.write(f" - Duration (ms): {features['duration_ms']}")
        else:
            st.write("No features found for this track.")

        # Back button
        if st.button("Back to Search"):
            st.session_state.page = 'search'
            st.rerun()