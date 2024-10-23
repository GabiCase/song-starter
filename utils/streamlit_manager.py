import streamlit as st
from utils.functions import get_track_id_spotify
def get_user_input_streamlit():
    st.title("Encuentra el Track ID de tu Canción Favorita")

    # Entrada para el nombre de la canción y el artista
    input_song = st.text_input("Introduce el nombre de la canción que te inspira:")
    input_artist = st.text_input("Introduce el nombre del artista que la interpreta:")

    return input_song, input_artist

def handle_input_search_streamlit(input_song, input_artist):
    song= "{input_artist} {input_song}"
    if input_song and input_artist:
        # Llamamos a la función para obtener el track ID
        track_id = get_track_id_spotify(song)
        st.success(f"Canción encontrada")
        return track_id
        
    else:
        # Muestra un mensaje de error y vuelve a mostrar los inputs
        st.error("Por favor, introduce tanto el nombre de la canción como el del artista.")
        input_song = st.text_input("Introduce el nombre de la canción que te inspira:")
        input_artist = st.text_input("Introduce el nombre del artista que la interpreta:")