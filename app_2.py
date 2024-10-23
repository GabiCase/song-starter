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
    input_track_id=0
    # Inicializar Pinecone y configuración del índice
    pc = initialize_pinecone()
    index = setup_index(pc)
    index_name= pc.list_indexes().indexes[0].name
    
    # Título principal de la app
    st.title("🎵 Song-starter: Inspiración musical 🎵")
    st.markdown("**Crea nuevas canciones inspiradas en tus referencias**")

    # Recoger la entrada del usuario (canción y artista)
    with st.sidebar:
        st.header("🔍 Buscar canción")
        input_song, input_artist = get_user_input_streamlit()

        if st.button('Buscar'):
            input_track_id = handle_input_search_streamlit(input_song, input_artist)
        else:
            st.write("Introduce el nombre de una canción y artista para empezar.")
        
    
    if input_track_id != 0:

        input_song , input_artist = get_song_and_artist(input_track_id)

        # Crear DataFrame inicial con información de la canción
        input_data = pd.DataFrame({'id': [input_track_id], 'name': input_song, 'artists': input_artist})

        st.markdown(f"### 🎶 Si te gusta **{input_song}** de **{input_artist}**")
        st.write("### Puedo recomendarte:")

        with st.spinner('Buscando recomendaciones...'):
            # Obtener canciones similares (puede tardar un poco)
            tracks_data = get_similar_songs(input_track_id, 4)

        cols = st.columns(2)

        # Lista para almacenar los nombres de las canciones y artistas
        names = [f"**{track['name']}** de {track['artists']}" for track in tracks_data]

        # Muestra cada canción y artista uno por uno con un retraso en columnas
        for i, name in enumerate(names):
            with cols[i % 2]:
                st.markdown(name)  # Muestra el nombre y artista en negrita
            time.sleep(2)  # Pausa de 2 segundos antes de mostrar la siguiente

        st.write("")
        st.write("")

        # Crear un contenedor para el mensaje temporal
        message_placeholder = st.empty()

        # Mostrar el mensaje inicial en el contenedor
        message_placeholder.markdown("### 🔍 Qué podemos encontrar de esta canción...")

        tracks_data = pd.DataFrame(tracks_data)

        # Filtrar IDs existentes en Pinecone
        filter_existing_vector_id(index, tracks_data) 
        tracks_data.reset_index(drop=True, inplace=True)

        # Leer archivo CSV temporal
        csv = pd.read_csv('mi_archivo_temporal.csv')
        df = check_track_csv(tracks_data, csv)
        df.reset_index(drop=True, inplace=True)

        # Mostrar mensaje mientras se buscan acordes
        with st.spinner('🎶 Buscando acordes...'):
            # Agregar información sobre acordes
            add_info_list_df(input_data, get_chords_chat, 'chords')  # Acordes de la canción de entrada
            add_info_list_df(df, get_chords_chat, 'chords')          # Acordes para canciones similares

        # Mostrar mensaje mientras se buscan los instrumentos
        with st.spinner('🎸 Identificando instrumentos...'):
            # Agregar información sobre instrumentos
            add_info_list_df(input_data, get_instruments_chat, 'instruments')  # Instrumentos de la canción de entrada
            add_info_list_df(df, get_instruments_chat, 'instruments') 

        # Mostrar spinner con un mensaje personalizado mientras se obtienen los datos
        with st.spinner('🧐 A ver de qué trata...'):
            # Obtener características y análisis de audio para la canción de entrada
            get_audio_features(input_data)
            get_audio_analysis(input_data)
            input_data, theme = insert_lyrics_db(input_data)

        # Eliminar el mensaje al finalizar el proceso
        message_placeholder.empty()

        # Añadir espacio para mejorar la separación visual
        st.write("")
        st.write("")
        # Mensaje personalizado sobre el tema de interés
        st.markdown("### 🎵 Así que este es el tema que te interesa:")
        st.markdown(f"*{theme}...*")

        # Filtrar acordes no nulos de input_data
        input_data['chords'].dropna(inplace=True)
        input_data = input_data[input_data['chords'].apply(lambda x: x != [])]
        
        # Obtener características y análisis de audio para las canciones similares
        get_audio_features(df)
        get_audio_analysis(df)
        insert_lyrics_db(df) 

        # Preparar DataFrame vectorial
        df_vectorial = df.copy()
        df_vectorial['chords'].dropna(inplace=True)
        df_vectorial = df_vectorial[df_vectorial['chords'].apply(lambda x: x != [])]

        # Mapa de acordes
        chord_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 
            'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 
            'A#': 10, 'Bb': 10, 'B': 11
        }

        input_data['chords_1'] = input_data['chords'].apply(lambda x: x[:4]) 
        input_data['chords_2'] = input_data['chords'].apply(lambda x: x[4:])

        input_data['chords_1'] = input_data['chords_1'].apply(standardize_chord_symbols)
        input_data['chords_2'] = input_data['chords_2'].apply(standardize_chord_symbols)

        input_data['chords_1_numeric'] = input_data['chords_1'].apply(convert_chords_with_minor)
        input_data['chords_2_numeric'] = input_data['chords_2'].apply(convert_chords_with_minor)

        col_chords = ['chords_1_numeric', 'chords_2_numeric']

        # Aplicar la función a cada fila
        for col in col_chords:
            input_data[f'normal_{col}'] = input_data.apply(lambda row: transpose_chords_to_key(row, col), axis=1)

        for col in col_chords:
            input_data[f'diff_{col}'] = input_data.apply(lambda row: calc_diff(row[col]), axis=1)

        input_data.replace('\n', ' ', regex=True, inplace=True)

        df_vectorial['chords_1'] = df_vectorial['chords'].apply(lambda x: x[:4]) 
        df_vectorial['chords_2'] = df_vectorial['chords'].apply(lambda x: x[4:])
        df_vectorial.drop(columns=['chords'], inplace=True)

        # Nos ocupamos de bemoles o sostenidos para estandarizarlos
        df_vectorial['chords_1'] = df_vectorial['chords_1'].apply(standardize_chord_symbols)
        df_vectorial['chords_2'] = df_vectorial['chords_2'].apply(standardize_chord_symbols)

        # Convertimos los acordes a valores numéricos
        df_vectorial['chords_1_numeric'] = df_vectorial['chords_1'].apply(convert_chords_with_minor)
        df_vectorial['chords_2_numeric'] = df_vectorial['chords_2'].apply(convert_chords_with_minor)

        col_chords = ['chords_1_numeric', 'chords_2_numeric']

        # Aplicar la función a cada fila
        for col in col_chords:
            df_vectorial[f'normal_{col}'] = df_vectorial.apply(lambda row: transpose_chords_to_key(row, col), axis=1)

        for col in col_chords:
            df_vectorial[f'diff_{col}'] = df_vectorial.apply(lambda row: calc_diff(row[col]), axis=1)

        df_vectorial.replace('\n', ' ', regex=True, inplace=True)
        #elimino los saltos de línea en la letra y temática de la canción

        df_vectorial = df_vectorial.fillna(0)

        # Especificar las columnas que contienen listas (excepto 'embedding')
        columns_to_expand = ['instruments', 'chords_1', 'chords_2', 
                        'chords_1_numeric', 'chords_2_numeric', 
                        'normal_chords_1_numeric', 'normal_chords_2_numeric', 
                        'diff_chords_1_numeric', 'diff_chords_2_numeric']
        
        input_expanded = expand_and_remove_original_columns(input_data, columns_to_expand)

        #elimino las columnas que tengan solo valores nulos
        input_expanded = input_expanded.dropna(axis=1, how='all')

        df_vectorial_expanded = expand_and_remove_original_columns(df_vectorial, columns_to_expand)

        #elimino las columnas que tengan solo valores nulos
        df_vectorial_expanded = df_vectorial_expanded.dropna(axis=1, how='all')

        drop_columns_with_many_nulls(df_vectorial_expanded, 10)

        input_expanded=to_embbed_text_column(input_expanded)

        df_vectorial_expanded=to_embbed_text_column(df_vectorial_expanded)

        api_key = os.getenv("OPENAI_API_KEY")

        # Configurar la clave de API en OpenAI
        OpenAI.api_key = api_key

        # Convert text into vectors using embeddings
        client = OpenAI(
        api_key=api_key,
        )

        input_expanded['embedding'] = input_expanded['to_embbed_text'].apply(lambda x: client.embeddings.create(
        model="text-embedding-3-small",
        input=x
        ).data[0].embedding)

        input_expanded.drop(columns=['to_embbed_text'], inplace=True)

        #para cada fila del dataframe, sacar el vector de embedding de la columna "to_embbed_text"
        df_vectorial_expanded['embedding'] = df_vectorial_expanded['to_embbed_text'].apply(lambda x: client.embeddings.create(
        model="text-embedding-3-small",
        input=x
        ).data[0].embedding)

        df_vectorial_expanded.drop(columns=['to_embbed_text'], inplace=True)

        df_vectorial_expanded = df_vectorial_expanded.rename(columns={'embedding': 'values'})

        # Procesar el DataFrame
        df_vectorial_expanded = create_metadata_column(df_vectorial_expanded)
        df_vectorial_expanded = clean_metadata_column(df_vectorial_expanded)
        df_para_pinecone = create_dataframe_for_pinecone(df_vectorial_expanded)
        insert_into_pinecone(df_para_pinecone, index)

        # Get the response from the query
        response = realizar_consulta(client, input_expanded, index, top_k=5)

        # Extract the matches from the response
        response_matches = response['matches']

        # Initialize lists for details and tempos
        details = []
        tempos = []

        # Process the response matches
        for match in response_matches:
            # Extract metadata
            instruments_1 = match['metadata'].get('instruments_1', 'Unknown')
            instruments_2 = match['metadata'].get('instruments_2', 'Unknown')
            tempo = match['metadata'].get('tempo', 0)
            speechiness = match['metadata'].get('speechiness', 0)
            loudness = match['metadata'].get('loudness', 0)

            # Append details in the desired format to be used later
            details.append(f"Intruments: {instruments_1} {instruments_2}, Tempo: {tempo}, Speechiness: {speechiness}, Loudness: {loudness}")

            # Collect the tempo values for immediate processing
            tempos.append(float(tempo))

        # Get the minimum and maximum tempo values (e.g., the first two tempos)
        tempo_min = min(tempos)  # Smallest of the first two tempos
        tempo_max = max(tempos)  # Largest of the first two tempos

        # Display the tempo suggestion (you can now use the details list later as needed)
        st.write(f"Se me ocurre que el tempo esté entre {tempo_min:.0f} y {tempo_max:.0f}.")

        diff_chord_2= input_data['diff_chords_2_numeric'].iloc[0]
        diff_chord_1= input_data['diff_chords_1_numeric'].iloc[0]

        chords_query = f"A song with a similar progession and distance (diff) in chords to {input_data['chords_1'][0]},y diff chords {diff_chord_2},{diff_chord_1}."
        theme_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=chords_query
        ).data[0].embedding

        chords_query = index.query(vector=theme_embedding, top_k=5, include_metadata=True)

        chords_query_ = chords_query['matches']

        chord_wheels=[]
        for match in chords_query_:
            chord_wheels.append(f"Chord wheel: {match['metadata']['chords_1_1']} {match['metadata']['chords_1_2']} {match['metadata']['chords_1_3']} {match['metadata']['chords_1_4']}, {match['metadata']['chords_2_1']} {match['metadata']['chords_2_2']} {match['metadata']['chords_2_3']} {match['metadata']['chords_2_4']}")

        # Crear el embedding de la nueva query
        lyrics_query = f"Give me a similar theme song to {input_data['theme'][0]}"
        theme_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=lyrics_query
        ).data[0].embedding

        lyrics_result = index.query(vector=theme_embedding, top_k=5, include_metadata=True)

        lyrics_matches = lyrics_result['matches']

        lyrics_list = []  # Crear una lista vacía para almacenar los textos
        

        for match in lyrics_matches:
            # Agregar cada texto generado a la lista
            lyrics_list.append(f"Lyrics: {match['metadata']['theme']}")
        lyrics_list.append(df['theme'])
        
        for match in lyrics_matches:
            # Agregar cada texto generado a la lista
            print(f"Lyrics: {match['metadata']['theme']} {match['id']} Score: {match['score']}")

        st.markdown("### 🎤 Nueva canción basada en tus preferencias")
        nueva_cancion = create_song(lyrics_list, chord_wheels, details)

        st.markdown(nueva_cancion)

    
if __name__ == '__main__':
    main()
