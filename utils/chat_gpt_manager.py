from openai import OpenAI
import os

from dotenv import load_dotenv
load_dotenv() 

api_key = os.getenv("OPENAI_API_KEY")

# Configurar la clave de API en OpenAI
OpenAI.api_key = api_key

client = OpenAI(
 api_key=api_key,
)


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


### Verso 1

def create_song(lyrics_list, chord_wheels, details, model="gpt-4o", temperature=1, max_tokens=2048):
    response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "Eres un asistente experto en escribir canciones. A continuación, te proporcionaré letras de canciones, ruedas de acordes, instrumentos a usar, tempo y potencia de la canción. Usa estos detalles con libertad para crear una nueva canción corta con una estructura clara y bien definida, que se exprese en el mismo tono y transmita una actitud similar."
        },
        {
            "role": "user",
            "content": f"""
    Basándote en: {lyrics_list},{chord_wheels}** y **{details}**, escribe una canción siguiendo esta estructura y formateándola en **markdown** de manera que quede visualmente clara y atractiva:

    # Título de la canción: 
    Proporciona un título basado en las temáticas dadas y las letras.

    **Tempo:** 
    **Instrumentos principales:** 

    ### Letra y acordes:

    1. **Verso 1**: Un breve verso de 4 líneas para introducir el tema.
    2. **Coro**: Un coro de 4 líneas que resuma el sentimiento principal de la canción.
    3. **Verso 2**: Otro verso corto de 4 líneas que desarrolle el tema o lo varíe.
    4. **Coro final**: Repite el coro y modifícalo ligeramente para cerrar la canción.

    **Nota**: La canción debe ser corta, enfocada en transmitir una idea clara y contundente en menos de 16 líneas. El tempo, los acordes y los instrumentos deben reflejar el estilo general que se basa en {lyrics_list} y {chord_wheels}.
    """
        }
    ],
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "text"
    }
    )

    return response.choices[0].message.content