import requests
from utils.chat_gpt_manager import chat_gpt_theme
import os


# Función para manejar el tiempo límite
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

# Definición de la función para obtener las primeras cuatro líneas
def get_first_lines(artist, title):
    url = f'https://api.lyrics.ovh/v1/{artist}/{title}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        lyrics = data['lyrics']
        first_lines = '\n'.join(lyrics.splitlines()[:8])
        return first_lines
    else:
        return None


def insert_lyrics_db(df):
    import time
    df['theme'] = None  # Inicializar la columna 'theme' como None

    for index, row in df.iterrows():
        try:
            # Establecer la alarma para que se active en 5 segundos
            time.sleep(5)

            # Intentar obtener el tema de la canción desde la API externa
            theme = get_first_lines(row['artists'], row['name'])
            if theme is None:  # Si la API no retorna un tema válido
                raise TimeoutException()

            df.at[index, 'theme'] = theme

        except TimeoutException:
            print(f"Timeout o fallo! Usando GPT para obtener el tema para {row['artists']} - {row['name']}")
            try:
                # Si hubo timeout o fallo, usar la función alternativa para obtener el tema
                theme = chat_gpt_theme(row['name'], row['artists'])
                df.at[index, 'theme'] = theme
            except Exception as e:
                print(f"Error al obtener tema con GPT para {row['artists']} - {row['name']}: {e}")
                df.at[index, 'theme'] = "No se pudo obtener el tema"

        except Exception as e:
            print(f"Error al obtener el tema para {row['artists']} - {row['name']}: {e}")
            df.at[index, 'theme'] = "No se pudo obtener el tema"


    return df, theme



