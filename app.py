import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from openai import OpenAI
import streamlit as st
import spotipy 

# Cargar variables de entorno desde un archivo .env
load_dotenv()

# Importar funciones de gestión
from utils.chat_gpt_manager import (
    get_instruments_chat,
    get_chords_chat
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
    get_user_input,
    handle_input_search,
    calc_diff,
    create_metadata_column,
    insert_into_pinecone,
    create_dataframe_for_pinecone

)
from utils.pinecone_manager import (
    initialize_pinecone, 
    setup_index
)
from utils.lyrics_manager import insert_lyrics_db
from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone


# app.py

def main():
    # Inicializar Pinecone y configuración del índice
    pc = initialize_pinecone()
    index = setup_index(pc)
    index_name= pc.list_indexes().indexes[0].name
    
    # Obtener entrada del usuario
    input_song, input_artist = get_user_input()
    input_track_id = handle_input_search(input_song, input_artist)
    
    # Obtener información de la canción y artista
    input_song, input_artist = get_song_and_artist(input_track_id)

    # Crear DataFrame inicial con información de la canción
    input_data = pd.DataFrame({'id': [input_track_id], 'name': input_song, 'artists': input_artist})

    # Obtener canciones similares
    tracks_data = get_similar_songs(input_track_id, 5)
    tracks_data = pd.DataFrame(tracks_data)
    
    # Filtrar IDs existentes en Pinecone
    filter_existing_vector_id(index, tracks_data) 
    tracks_data.reset_index(drop=True, inplace=True)
    
    # Leer archivo CSV temporal
    csv = pd.read_csv('mi_archivo_temporal.csv')
    df = check_track_csv(tracks_data, csv)
    df.reset_index(drop=True, inplace=True)
    
    # Agregar información sobre acordes e instrumentos
    # Agregar información sobre acordes
    add_info_list_df(input_data, get_chords_chat, 'chords')  # Obtener acordes para la canción de entrada
    add_info_list_df(df, get_chords_chat, 'chords')          # Obtener acordes para las canciones similares

    # Agregar información sobre instrumentos
    add_info_list_df(input_data, get_instruments_chat, 'instruments')  # Obtener instrumentos para la canción de entrada
    add_info_list_df(df, get_instruments_chat, 'instruments')          # Obtener instrumentos para las canciones similares

    # Obtener características y análisis de audio para la canción de entrada
    get_audio_features(input_data)
    get_audio_analysis(input_data)
    insert_lyrics_db(input_data)

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

        
        # Verificar si el lote no está vacío
        if not batch.empty:
            index.upsert_from_dataframe(batch)

    

if __name__ == '__main__':
    main()
