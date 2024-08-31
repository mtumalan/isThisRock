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

@app.get("/genre")
def get_songs_by_genre(genre: str = Query(..., description="The genre of the songs to search for")):
    # Construct the query to search by genre
    query = f'genre:"{genre}"'
    
    tracks = []
    
    # Fetch up to 100 tracks (Spotify API returns up to 50 tracks per request)
    limit = 50
    offset = 0
    
    while len(tracks) < 100:
        results = spotify.search(q=query, type='track', limit=limit, offset=offset)
        items = results['tracks']['items']
        
        if not items:
            break  # Stop if no more tracks are found
        
        for item in items:
            track_info = {
                'id': item['id'],
                'name': item['name'],
                'artists': [artist['name'] for artist in item['artists']],
            }
            tracks.append(track_info)
        
        offset += limit  # Move to the next batch of results

    return {"results": tracks[:100]}  # Return only the first 100 tracks