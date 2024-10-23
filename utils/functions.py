import re
from utils.chat_gpt_manager import get_chords_chat, get_instruments_chat
import json
import pandas as pd
import ast
import os
from utils.spotify_manager import get_track_id_spotify
import numpy as np





# Para extraer listas de acordes de la respuesta y asegurarnos de aplanar si es necesario
import re

# Función para extraer la lista de acordes
def extract_chat_list(respuesta_chat):
    pattern = r"\[([^\]]+)\]"
    listas_encontradas = re.findall(pattern, respuesta_chat)
    
    # Extraemos las listas y las convertimos a formato Python (quitando comillas y espacios)
    acordes = [lista.replace("'", "").split(", ") for lista in listas_encontradas]
    
    # Aplanamos la lista de acordes si es una lista de listas
    if len(acordes) > 0 and isinstance(acordes[0], list):
        acordes_planos = [acorde for sublist in acordes for acorde in sublist]
    else:
        acordes_planos = acordes  # Si ya es una lista simple, la dejamos tal cual
    
    return acordes_planos if acordes_planos else None

# Función para obtener la información y almacenarla en una columna

def add_info_list_df(df, function, column_name, max_retries=3):
    df[column_name] = ''  # Añade la columna vacía al DataFrame

    if len(df) > 0:
        for i in range(len(df)):
            titulo = df.at[i, 'name']
            artista = df.at[i, 'artists']
            intentos = 0
            while intentos < max_retries:
                try:
                    # Llamamos a la función que se pasa como argumento para obtener la información
                    respuesta = function(titulo, artista)
                    respuesta = extract_chat_list(respuesta)

                    # Verificamos la longitud de la respuesta dependiendo de la columna
                    if column_name == 'chords' and (respuesta is None or len(respuesta) < 8):
                        intentos += 1  # Incrementa los intentos si no es válida
                        print(f"Intento {intentos} fallido para '{titulo}' de '{artista}' (acordes insuficientes)")
                    elif column_name == 'instruments' and (respuesta is None or len(respuesta) < 2):
                        intentos += 1  # Incrementa los intentos si no es válida
                        print(f"Intento {intentos} fallido para '{titulo}' de '{artista}' (instrumentos insuficientes)")
                    else:
                        # Si la respuesta es válida, almacenamos la información procesada en el DataFrame
                        df.at[i, column_name] = respuesta
                        break  # Salimos del bucle si la respuesta es correcta

                except Exception as e:
                    intentos += 1
                    print(f"Error al obtener la información de la canción '{titulo}' de '{artista}' en el intento {intentos}: {e}")

            # Si se exceden los intentos y no hay respuesta válida, dejamos el campo vacío
            if intentos == max_retries:
                print(f"No se pudo obtener información válida para '{titulo}' de '{artista}' después de {max_retries} intentos")
                df.at[i, column_name] = ""

        # Reseteamos los índices del DataFrame después del proceso
        df.reset_index(drop=True, inplace=True)

    return df



def standardize_chord_symbols(chord):

    return [c.replace('♭', 'b') for c in chord]
   
def transpose_chords_to_key(row, col_acordes):
    return [chord - row['key'] for chord in row[col_acordes]]


# Mapeo de notas a números (la raíz de los acordes)
chord_map = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 
    'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 
    'A#': 10, 'Bb': 10, 'B': 11
}

def convert_chords_with_minor(chords):
    chord_numbers = []
    for chord in chords:
        # Extraer la raíz del acorde usando una expresión regular
        match = re.match(r'^[A-G][#b]?', chord)  # Coincide con la raíz del acorde (ej: C, C#, Db)
        if match:
            root = match.group(0)  # Obtener la raíz del acorde
            # Determinar si es acorde menor (termina en 'm')
            chord_number = chord_map.get(root, 0)  # Obtener el número de la raíz, por defecto 0
            if chord.endswith('m'):
                chord_numbers.append(chord_number + 12)  # Acorde menor -> suma 12
            else:
                chord_numbers.append(chord_number)  # Acorde mayor o con extensiones que ignoramos
        else:
            chord_numbers.append(0)  # Si no coincide con un acorde válido
    return chord_numbers


def fill_chords(df):
    # Función interna para rellenar los acordes
    def repeat_chords(chords):
        # Verificar si los acordes son una lista válida
        if isinstance(chords, list) and len(chords) < 8:
            # Repetir los acordes hasta completar 8
            while len(chords) < 8:
                chords += chords[:8 - len(chords)]  # Añadir tantos acordes como faltan
        return chords
    
    # Aplicar la función de relleno en cada fila de la columna 'chords'
    df['chords'] = df['chords'].apply(repeat_chords)
    
    return df


def check_track_csv(df, csv):
    # Extraer los 'id' del CSV como un conjunto para comparación rápida
    csv_ids = set(csv['id'])
    
    # Filtrar el DataFrame eliminando las filas donde el 'id' coincida con alguno en 'csv_ids'
    df = df[~df['id'].isin(csv_ids)]
    
    return df

def clean_metadata(metadata):
    cleaned_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            cleaned_metadata[key] = value
        elif isinstance(value, (list, dict)):
            # Convierte listas o diccionarios a una cadena JSON
            cleaned_metadata[key] = json.dumps(value)
        else:
            cleaned_metadata[key] = str(value)  # Convierte cualquier otro tipo a cadena
    return cleaned_metadata

def save_non_existing_vector_id(index, df):
    non_existent_ids = []
    # Iterar sobre cada fila del DataFrame
    for i, row in df.iterrows():
        id_vector = row['id']  # Obtener el ID del vector

        # Verificar si existe el vector con el ID
        result = index.fetch(ids=[id_vector])
        
        if id_vector in result['vectors']:
            print(f"El vector con ID {id_vector} existe en Pinecone.")
        else:
            non_existent_ids.append(id_vector)
    return non_existent_ids

def filter_existing_vector_id(index, tracks_data):
    # Primero, recopilamos los IDs que existen en Pinecone
    existing_ids = []
    for i, row in tracks_data.iterrows():
        id_vector = row['id']  # Obtener el ID del vector
        try:
            # Verificar si existe el vector con el ID
            result = index.fetch(ids=[id_vector])

            if id_vector in result['vectors']:
                existing_ids.append(i)  # Guardar el índice de la fila a eliminar
        except Exception as e:
            print(f"Esta canción ya está en pinecone.  ID: {row['name']} - {row['artists']}: {e}")
    # Ahora eliminamos las filas que existen en Pinecone
    tracks_data.drop(existing_ids, inplace=True)

    # Función para dividir listas en nuevas columnas
def expand_list_columns(df, columns_to_expand):
    for column in columns_to_expand:
        # Expandir la lista en nuevas columnas
        list_df = df[column].apply(pd.Series)
        # Renombrar columnas
        list_df.columns = [f'{column}_{i+1}' for i in range(list_df.shape[1])]
        # Unir las nuevas columnas al DataFrame original
        df = df.join(list_df)
        # Eliminar la columna original
        df = df.drop(columns=[column])
    return df

def to_embbed_text_column(df):
    # Crear una nueva columna llamada 'to_embbed_text' donde se concatenarán el nombre de las columnas con sus valores,
    # excluyendo la columna 'id'
    df['to_embbed_text'] = df.apply(lambda row: ' '.join([f"{col}: {row[col]}" for col in df.columns if col != 'id']), axis=1)
    return df


def drop_columns_with_many_nulls(df, threshold):
    """
    Elimina columnas del DataFrame que tienen más de `threshold` valores nulos.
    
    Args:
    df (pd.DataFrame): El DataFrame original.
    threshold (int): El número máximo permitido de valores nulos por columna.
    
    Returns:
    pd.DataFrame: El DataFrame con las columnas eliminadas.
    """
    # Identificar las columnas que tienen más de 'threshold' valores nulos
    cols_to_drop = df.columns[df.isnull().sum() > threshold]
    
    # Eliminar esas columnas
    df = df.drop(columns=cols_to_drop)
    
    return df

def expand_and_remove_original_columns(df, column_names):
    """
    Expande las columnas en el DataFrame df según las columnas especificadas en column_names,
    convierte las cadenas de texto que representan listas en listas reales y elimina las columnas originales.
    Solo se expande hasta el número máximo de elementos presentes en cada columna.
    
    Args:
    df (pd.DataFrame): El DataFrame original.
    column_names (list): Lista de nombres de columnas que contienen listas de valores como cadenas de texto.
    
    Returns:
    pd.DataFrame: El DataFrame expandido con nuevas columnas.
    """
    for column in column_names:
        # Verificar si la columna existe en el DataFrame
        if column in df.columns:
            # Convertir las cadenas en listas usando ast.literal_eval si es necesario
            df[column] = df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            
            # Encontrar el máximo número de elementos en las listas de esta columna
            max_len = df[column].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
            
            # Expandir las listas en nuevas columnas sin generar columnas extra con NaN
            expanded = df[column].apply(lambda x: pd.Series(x) if isinstance(x, list) else pd.Series([None]*max_len))
            
            # Renombrar las columnas resultantes
            expanded.columns = [f"{column}_{i+1}" for i in range(expanded.shape[1])]
            
            # Concatenar las nuevas columnas al DataFrame original
            df = pd.concat([df, expanded], axis=1)
            
            # Eliminar la columna original
            df.drop(columns=[column], inplace=True)
    
    return df


def get_user_input():
    # Entrada para el nombre de la canción y el artista
    input_song = input("Introduce el nombre de la canción que te inspira:")
    input_artist = input("Introduce el nombre del artista que la interpreta:")

    return input_song, input_artist

def handle_input_search(input_song, input_artist):
    song= f"{input_artist} {input_song}"
    if input_song and input_artist:
        # Llamamos a la función para obtener el track ID
        track_id = get_track_id_spotify(song)
        return track_id
    else:
        # Muestra un mensaje de error y vuelve a mostrar los inputs
        print("Por favor, introduce tanto el nombre de la canción como el del artista.")
        input_song = input("Introduce el nombre de la canción que te inspira:")
        input_artist = input("Introduce el nombre del artista que la interpreta:")

def calc_diff(acordes):
    return np.diff(acordes).tolist()

def export_unique_rows(df, csv_file):
    # Verificar si el archivo ya existe
    if os.path.exists(csv_file):
        # Leer el CSV existente
        df_existing = pd.read_csv(csv_file)

        # Concatenar los nuevos datos con los existentes
        df_combined = pd.concat([df_existing, df])
    else:
        # Si el archivo no existe, los datos combinados son solo los nuevos
        df_combined = df

    # Convertir las listas en cadenas para poder eliminar duplicados
    df_combined = df_combined.applymap(lambda x: str(x) if isinstance(x, list) else x)

    # Eliminar filas duplicadas
    df_combined.drop_duplicates(inplace=True)

    # Convertir las cadenas de vuelta a listas (si es necesario)
    df_combined = df_combined.applymap(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x)

    # Guardar los datos combinados en el archivo CSV
    df_combined.to_csv(csv_file, index=False)

def create_metadata_column(df):
    """Crear un diccionario de metadatos a partir de las columnas restantes."""
    metadata_columns = [col for col in df.columns if col not in ['id', 'values']]
    df['metadata'] = df[metadata_columns].apply(lambda x: x.to_dict(), axis=1)
    return df

def clean_metadata_column(df):
    """Aplicar la función de limpieza a la columna de metadatos."""
    df['metadata'] = df['metadata'].apply(clean_metadata)
    return df

def create_dataframe_for_pinecone(df):
    """Crear un nuevo DataFrame con la estructura correcta para Pinecone."""
    return pd.DataFrame({
        'id': df['id'],
        'values': df['values'],
        'metadata': df['metadata']
    })

def insert_into_pinecone(df, index, batch_size=100):
    """Insertar datos en Pinecone en lotes especificados."""
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        
        # Verificar si el lote no está vacío
        if not batch.empty:
            index.upsert_from_dataframe(batch)


def realizar_consulta(client,df, index, top_k=5):
    # Crear el embedding de la nueva query
    input_query = f"{df['embedding']}"
    input_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=input_query
    ).data[0].embedding

    # Realizar la consulta al índice
    response = index.query(vector=input_embedding, top_k=top_k, include_metadata=True)
    
    # Imprimir y devolver el resultado
    print("Resultado de la consulta:", response)
    return response