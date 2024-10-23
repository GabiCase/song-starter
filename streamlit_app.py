import pandas as pd
import streamlit as st
from utils.chat_gpt_manager import get_instruments_chat, get_chords_chat
from utils.spotify_manager import get_similar_songs, get_song_and_artist
from utils.functions import (
    add_info_list_df, get_user_input, handle_input_search, calc_diff
)
from utils.pinecone_manager import initialize_pinecone, setup_index

from utils.streamlit_manager import (
    get_user_input_streamlit,
    handle_input_search_streamlit
)

def main():
    st.title("Music Analyzer App")

    # Inicializar Pinecone y configuración del índice
    pc = initialize_pinecone()
    index = setup_index(pc)

    # Obtener entrada del usuario
    input_song, input_artist = get_user_input_streamlit()

    if st.button("Buscar canción"):
        input_track_id = handle_input_search_streamlit(input_song, input_artist)
        
        # Mostrar canción y artista seleccionados
        st.write(f"Has seleccionado la canción: {input_song} por {input_artist}")
        
        # Obtener información de la canción y artista
        input_song, input_artist = get_song_and_artist(input_track_id)

        # Crear DataFrame inicial con información de la canción
        input_data = pd.DataFrame({'id': [input_track_id], 'name': input_song, 'artists': input_artist})

        # Obtener canciones similares
        st.write("Obteniendo canciones similares...")
        tracks_data = get_similar_songs(input_track_id, 5)
        tracks_data = pd.DataFrame(tracks_data)
        st.write("Canciones similares encontradas:")
        st.dataframe(tracks_data)

        # Agregar información sobre acordes e instrumentos
        add_info_list_df(input_data, get_chords_chat, 'chords')
        add_info_list_df(input_data, get_instruments_chat, 'instruments')

        st.write("Acordes e instrumentos de la canción seleccionada:")
        st.dataframe(input_data[['chords', 'instruments']])

        # Cálculo de diferencia de acordes
        input_data['diff_chords'] = input_data.apply(lambda row: calc_diff(row['chords']), axis=1)
        st.write("Diferencias entre acordes:")
        st.dataframe(input_data[['chords', 'diff_chords']])

if __name__ == '__main__':
    main()
