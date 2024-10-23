import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from openai import OpenAI
import streamlit as st
import spotipy 
import time

# Cargar variables de entorno desde un archivo .env
load_dotenv()

# Importar funciones de gestión
from utils.chat_gpt_manager import (
    get_instruments_chat,
    get_chords_chat,
    create_song
)
from utils.spotify_manager import (
    get_similar_songs,
    get_track_id_spotify,
    get_audio_analysis,
    get_audio_features,
    get_song_and_artist
)
from utils.functions import (
    add_info_list_df,
    convert_chords_with_minor,
    standardize_chord_symbols,
    transpose_chords_to_key,
    check_track_csv,
    clean_metadata_column,
    filter_existing_vector_id,
    to_embbed_text_column,
    expand_and_remove_original_columns,
    drop_columns_with_many_nulls,
    calc_diff,
    create_metadata_column,
    insert_into_pinecone,
    create_dataframe_for_pinecone,
    realizar_consulta

)
from utils.pinecone_manager import (
    initialize_pinecone, 
    setup_index
)

from utils.streamlit_manager import (
    get_user_input_streamlit,
    handle_input_search_streamlit
)

from utils.lyrics_manager import insert_lyrics_db

from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone


# app.py

def main():
    input_track_id = 0
    # Inicializar Pinecone y configuración del índice
    pc = initialize_pinecone()
    index = setup_index(pc)
    index_name = pc.list_indexes().indexes[0].name
    
    st.title("Song-starter")
    # Llamar a la función para obtener la entrada del usuario
    input_song, input_artist = get_user_input_streamlit()

    # Añadir un botón para realizar la búsqueda
    if st.button('Buscar'):
        input_track_id = handle_input_search_streamlit(input_song, input_artist)
        
    if input_track_id != 0:
        with st.expander("Detalles de la canción seleccionada", expanded=True):
            input_song, input_artist = get_song_and_artist(input_track_id)
            st.markdown(f"**Canción**: {input_song} \n\n **Artista**: {input_artist}")

        st.markdown(f"""
            <p style='font-size:18px;'> Si te gusta <strong>{input_song}</strong> de <strong>{input_artist}</strong>, puedo recomendarte algunas canciones similares.</p>
            """, unsafe_allow_html=True)

        with st.spinner('Buscando recomendaciones...'):
            tracks_data = get_similar_songs(input_track_id, 4)

        # Lista para almacenar los nombres de las canciones y artistas
        names = [f"**{track['name']}** de {track['artists']}" for track in tracks_data]

        # Mostrar cada canción con un retraso de 2 segundos
        for name in names:
            st.markdown(name)
            time.sleep(2)

        st.write("")

        st.markdown("*Vamos a investigar más detalles de tu canción:*")

        tracks_data = pd.DataFrame(tracks_data)

        # Filtrar IDs existentes en Pinecone
        filter_existing_vector_id(index, tracks_data) 
        tracks_data.reset_index(drop=True, inplace=True)

        # Progreso al cargar el CSV
        progress_bar = st.progress(0)
        with st.spinner('Leyendo archivo CSV...'):
            csv = pd.read_csv('mi_archivo_temporal.csv')
            progress_bar.progress(50)
            df = check_track_csv(tracks_data, csv)
            df.reset_index(drop=True, inplace=True)
            progress_bar.progress(100)

        st.write("")

        # Mostrar acorde e instrumentos con spinners
        with st.spinner('Buscando acordes e instrumentos...'):
            add_info_list_df(input_data, get_chords_chat, 'chords')
            add_info_list_df(df, get_chords_chat, 'chords')

            add_info_list_df(input_data, get_instruments_chat, 'instruments')
            add_info_list_df(df, get_instruments_chat, 'instruments')

        with st.spinner('Analizando características de audio y letra...'):
            get_audio_features(input_data)
            get_audio_analysis(input_data)
            input_data, theme = insert_lyrics_db(input_data)

        st.write(f"El tema de la canción es **{theme}**...")

        # Mostrar progreso en la búsqueda de acordes
        st.write("Procesando acordes y embebiendo texto...")
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        # El resto de la lógica de procesamiento de acordes y embebimiento
        input_expanded = process_input_data(input_data)
        df_vectorial_expanded = process_vectorial_data(df)

        # Insertar en Pinecone y realizar consulta
        insert_into_pinecone(df_vectorial_expanded, index)
        response = realizar_consulta(client, input_expanded, index, top_k=5)

        # Mostrar resultados en expanders
        with st.expander("Resultados de la consulta", expanded=True):
            st.write(f"El tempo sugerido está entre {tempo_min:.0f} y {tempo_max:.0f}.")
            for detail in details:
                st.markdown(detail)

        st.write("Creando una nueva canción basada en tus intereses...")
        nueva_cancion = create_song(lyrics_list, chord_wheels, details)

        st.markdown(nueva_cancion)

if __name__ == '__main__':
    main()
