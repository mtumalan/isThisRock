from fastapi import FastAPI
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
from fastapi.params import Query

load_dotenv()

app = FastAPI()

spotify = Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
))

@app.get("/search")
def search_song(q: str = Query(..., description="The name of the song to search for")):
    results = spotify.search(q=q, type='track', limit=10)
    tracks = []
    for item in results['tracks']['items']:
        track_info = {
            'id': item['id'],
            'name': item['name'],
            'artists': [artist['name'] for artist in item['artists']],
            'album': item['album']['name'],
            'release_date': item['album']['release_date'],
            'image_url': item['album']['images'][0]['url'] if item['album']['images'] else None
        }
        tracks.append(track_info)
    return {"results": tracks}

@app.get("/features")
def get_features(track_id: str):
    features = spotify.audio_features(track_id)[0]
    return features