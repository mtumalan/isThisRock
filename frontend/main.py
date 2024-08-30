import streamlit as st
import requests

st.title("Is this Rock?")

st.write("Trained with Spotify API data, this model predicts whether a song is Rock or not. Try it out!")

song_name = st.text_input("Enter a song")

if song_name:
    # Call the backend API
    response = requests.get(f"http://api:8000/search", params={"q": song_name})
    results = response.json()

    if results["results"]:
        st.write("Top 10 search results:")
        for track in results["results"]:
            st.write(f"**{track['name']}** by {', '.join(track['artists'])}")
            st.write(f"Album: {track['album']}")
            st.write(f"Release Date: {track['release_date']}")
            if track['image_url']:
                st.image(track['image_url'], width=300)
            st.write("---")
    else:
        st.write("No results found")