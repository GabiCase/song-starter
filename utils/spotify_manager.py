import spotipy
import time
import os
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()  

# Spotify API 
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")


#Initialize SpotiPy with user credentials
SP = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID,
                                                           client_secret=CLIENT_SECRET))


def get_track_id_spotify(name_string):
    result= SP.search(q=f'{name_string}', limit=1)
    track=result['tracks']['items'][0]

    return track['id']

def get_song_and_artist(id):
    """
    Extrae el nombre de la canción y el artista de la respuesta de la API de Spotify.

    Esta función recibe el ID de una canción y devuelve un tuple con el nombre de la
    canción y el nombre del artista.

    Args:
        id (str): El ID de la canción en Spotify.

    Returns:
        tuple: Un tuple con el nombre de la canción y el nombre del artista.
    """
    # Realiza la búsqueda en Spotify usando el ID de la canción
    spotify_response = SP.track(id)

    # Extrae el nombre de la canción
    song_name = spotify_response.get("name")  # "Miedo"
    
    # Extrae el nombre del primer artista (si existe)
    artist_name = spotify_response["artists"][0].get("name") if spotify_response["artists"] else None  # "M-Clan"

    return song_name, artist_name  # Devuelve como un tuple



def get_similar_songs(track_id, limit=20):
    recommendations = SP.recommendations(seed_tracks=[track_id], limit=limit)
    tracks_data = []
    for track in recommendations['tracks']:
        time.sleep(2)
        artist_names = [artist['name'] for artist in track['artists']]
        track_info = {
            'id': track['id'],
            'name': track['name'],
            'artists': ', '.join(artist_names)
        }
        tracks_data.append(track_info)

    return tracks_data


def get_audio_analysis(df):

    keys = []  # Lista temporal para los valores de key
    modes = []  # Lista temporal para los valores de mode
    tempos = []  # Lista temporal para los valores de tempo
    loudness_values = []  # Lista temporal para los valores de loudness

    for _, row in df.iterrows():  # Iterar sobre cada fila del DataFrame
        time.sleep(1.5)
        track_id = row['id']  # Obtener el ID de la canción
        result = SP.audio_analysis(track_id)  # Obtener el análisis de audio de Spotify
        
        # Extraer key, mode y tempo
        key = result['track']['key']
        mode = result['track']['mode']
        tempo = result['track']['tempo']
        loudness = result['track']['loudness']

        # Guardar los valores en listas temporales
    
        keys.append(key)
        modes.append(mode)
        tempos.append(tempo)
        loudness_values.append(loudness)

    # Asignar las columnas al DataFrame una vez terminado el bucle

    df['key'] = keys
    df['mode'] = modes
    df['tempo'] = tempos
    df['loudness'] = loudness_values

    return df


def get_audio_features(df):
    # Create empty lists to hold the audio feature values
    speechiness_values = []
    instrumentalness_values = []

    # Iterate over each row of the DataFrame
    for _, row in df.iterrows():
        time.sleep(1.5)  # Sleep to avoid hitting rate limits
        track_id = row['id']  # Get the track ID
        
        # Get the audio features from Spotify
        result = SP.audio_features(track_id)
        
        # Check if result is not None and contains data
        if result and result[0]:  # result is a list of dictionaries
            features = result[0]  # Get the first (and only) dictionary
            # Extract speechiness and instrumentalness
            speechiness_values.append(features['speechiness'])
            instrumentalness_values.append(features['instrumentalness'])
        else:
            # Append None if no data is available
            speechiness_values.append(None)
            instrumentalness_values.append(None)

    # Assign the lists to the DataFrame columns
    df['speechiness'] = speechiness_values
    df['instrumentalness'] = instrumentalness_values

    return df
