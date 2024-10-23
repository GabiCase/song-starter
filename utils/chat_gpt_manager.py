from openai import OpenAI
import os

from dotenv import load_dotenv
load_dotenv() 

api_key = os.getenv("OPENAI_API_KEY")

# Configurar la clave de API en OpenAI
OpenAI.api_key = api_key


def get_chords_chat(cancion, artista):
    # Inicializa el cliente con la clave API desde las variables de entorno
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY") 
    )

    # Define el prompt con la estructura deseada
    prompt = f"Give me two lists, each with 4 elements, in Python format like this example ['chord', 'chord', 'chord', 'chord'], the chorus chord progressions in the song '{cancion}' by {artista}."

    # Realiza la solicitud a la API
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o-mini",  # Utilizamos el modelo adecuado
    )

    # Retorna el contenido de la respuesta
    return chat_completion.choices[0].message.content



def get_instruments_chat(cancion, artista):
    # Inicializa el cliente con la clave API desde las variables de entorno
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY") 
    )

    # Define el prompt con la estructura deseada
    prompt = f"Give me in a list of python format as this example ['piano', 'guitar', 'synthesizer'] only a list with the two first music instruments, digital or analog, in the song {cancion} by {artista}"

    # Realiza la solicitud a la API
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o-mini",  # Utilizamos el modelo adecuado
    )

    # Retorna el contenido de la respuesta
    return chat_completion.choices[0].message.content


def chat_gpt_theme(cancion, artista):
    # Inicializa el cliente con la clave API desde las variables de entorno
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY") 
    )

    # Define el prompt con la estructura deseada
    prompt = f"Tell me in three or four sentences the theme of the lyrics of the song {cancion} by {artista}"

    # Realiza la solicitud a la API
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o-mini",  # Utilizamos el modelo adecuado
    )

    # Retorna el contenido de la respuesta
    return chat_completion.choices[0].message.content