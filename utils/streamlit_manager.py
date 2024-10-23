import streamlit as st
from utils.functions import get_track_id_spotify
import time


def get_user_input_streamlit():
    

    # Entrada para el nombre de la canción y el artista
    input_song = st.text_input("Introduce el nombre de la canción que te inspira:",key="input_song")
    input_artist = st.text_input("Introduce el nombre del artista que la interpreta:",key="input_artist")

    return input_song, input_artist

def handle_input_search_streamlit(input_song, input_artist):
    if input_song and input_artist:
        # Combina el nombre del artista y de la canción
        song = f"{input_artist} {input_song}"
        
        # Llama a la función para obtener el track ID de Spotify
        track_id = get_track_id_spotify(song)
        
        # Si se encuentra el track ID, muestra un mensaje de éxito
        if track_id:
            message=st.success(f"Ya sé qué cancion es esa")
            time.sleep(2)
            message.empty()
            return track_id
        else:
            # Muestra un mensaje si no se encuentra el track ID
            st.error("No se encontró la canción. Verifica los datos e inténtalo nuevamente.")
            return 0
    else:
        # Muestra un mensaje de error si no se han introducido ambos datos
        st.error("Por favor, introduce tanto el nombre de la canción como el del artista.")
        return 0