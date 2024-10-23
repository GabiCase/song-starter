import pinecone
from dotenv import load_dotenv
import pandas as pd
import os
from pinecone import Pinecone, ServerlessSpec


# Carga las variables de entorno desde el archivo .env
load_dotenv()

from pinecone import ServerlessSpec

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT=  os.getenv('PINECONE_ENVIRONMENT')
# Initialize a client
pc = Pinecone(api_key=PINECONE_API_KEY)


index_name = "songstarter"
index = pinecone.Index(index_name, host=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT) 

def initialize_pinecone():
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc

# Función para verificar y crear el índice en Pinecone si no existe
def setup_index(pc, index_name="song-starter-index", dimension=1536, metric="cosine"):
    existing_indexes = pc.list_indexes()
    indexes = existing_indexes.get('indexes', [])

    # Verificar si el índice ya existe
    index_exists = any(index.get('name') == index_name for index in indexes)

    if not index_exists:
        # Crear el índice si no existe
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Especificar la región deseada
            ),
            deletion_protection="disabled"
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists, skipping creation.")
    
    return pc.Index(index_name)

def check_pinecone(name='songstarter'):
    
    # Verificar si la clave de API está disponible
    if PINECONE_API_KEY is None:
        raise ValueError("PINECONE_API_KEY no está configurada en las variables de entorno.")
        
    # Acceder al índice
    index = pc.Index(name)
    
    # Obtener y mostrar las estadísticas del índice
    stats = index.describe_index_stats()
    print(stats)



def check_songs_exist_pinecone(index, song_ids):
    """
    Verifica qué canciones ya existen en Pinecone.
    :param index: Índice de Pinecone.
    :param song_ids: Lista de IDs de canciones.
    :return: Un conjunto con los IDs de las canciones que ya existen en Pinecone.
    """
    try:
        # Usa index.fetch para obtener todas las canciones por sus IDs
        response = index.fetch(ids=song_ids)
        # Extraer los IDs de las canciones que ya existen
        existing_song_ids = set(response['vectors'].keys())  # Devuelve solo los IDs que están en Pinecone
        return existing_song_ids
    except Exception as e:
        return set()  # Devuelve un conjunto vacío
    

def insert_songs(index=index, df=''):
    """
    Inserta canciones en Pinecone, verificando antes si ya existen.
    :param index: Índice de Pinecone.
    :param df: DataFrame de pandas con columnas 'id' (IDs de canciones) y 'features' (vectores).
    """
    # Verifica si las canciones ya existen en Pinecone
    song_ids = df['id'].tolist()
    existing_song_ids = check_songs_exist_pinecone(index, song_ids)
    
    # Filtra el DataFrame para obtener solo las canciones que no están en Pinecone
    new_songs_df = df[~df['id'].isin(existing_song_ids)]

    # Inserta solo las canciones nuevas
    if not new_songs_df.empty:
        # Preparar los vectores para insertar en Pinecone
        vectors_to_upsert = [
            {'id': row['id'], 'values': row['features']}
            for _, row in new_songs_df.iterrows()
        ]
        try:
            # Inserta los nuevos vectores en Pinecone
            index.upsert(vectors_to_upsert)
            print(f"Se insertaron {len(vectors_to_upsert)} canciones nuevas en Pinecone.")
        except Exception as e:
            print(f"Error al insertar las canciones en Pinecone: {e}")
    else:
        print("No hay canciones nuevas para insertar.")

def insert_all_songs(index, songs):
    """
    Inserta todas las canciones en el índice de Pinecone.
    :param index: Índice de Pinecone.
    :param songs: Diccionario con ID de canciones como clave y características como valor.
    """
    # Convierte el diccionario en un DataFrame con columnas 'id' y 'features'
    df = pd.DataFrame([{'id': song_id, 'features': features} for song_id, features in songs.items()])
    
    # Inserta las canciones que no están en Pinecone
    insert_songs(index, df)
