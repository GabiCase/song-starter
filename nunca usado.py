def select_pitches(song):
    pitches_secuence=[]
    bars = song['bars']
    segments = song['segments']
    
    # Tomamos los primeros 16 elementos
    first_16_bars = bars[:16]
    
    # Inicializar una lista para guardar los pares barra-segmento para los primeros 16 "bars"
    bars_with_segments = []

    # Iterar solo sobre los primeros 16 "bars"
    for bar in first_16_bars:
        # Encontrar el segmento más lejano al inicio del "bar",  probé cercano e iba peor
        closest_segment = max(segments, key=lambda segment: abs(segment['start'] - bar['start']))
        
        # Añadir el par barra-segmento a la lista
        bars_with_segments.append({
            'bar': bar,
            'closest_segment': closest_segment
        })

    # Mostrar los resultados
    for pair in bars_with_segments:
       pitches_secuence.append(pair['closest_segment']['pitches'])
       
    return pitches_secuence






from music21 import chord

# Vector de ejemplo (ajusta según tus datos)
chroma_vector = [0.249,
  0.414,
  0.244,
  0.224,
  0.396,
  0.301,
  0.344,
  0.174,
  0.103,
  1.0,
  0.951,
  0.35]

# Mapeo de índices a notas
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Seleccionar los picos más altos (ajusta el umbral según sea necesario)
threshold = 0.5
pitches = [note_names[i] for i, v in enumerate(chroma_vector) if v > threshold]

# Crear un acorde a partir de los pitches seleccionados
detected_chord = chord.Chord(pitches)

# Obtener el nombre común del acorde
common_name = detected_chord.commonName
quality = detected_chord.quality  # Indica si es mayor, menor, aumentado, etc.

print(f"Acorde detectado: {common_name}")
print(f"Calidad del acorde: {quality}")
print(detected_chord)






import requests

def download_audio(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)
    print(f"Archivo descargado: {filename}")

# Ejemplo de URL con un archivo de audio (sustituye por tu URL)
url = 'https://p.scdn.co/mp3-preview/fad76e7d1907cc017f435b00b5878abbc527d819?cid=12d9a8c20d7d4e9c80a9dafe8204ab02'
filename = 'sample_audio.mp3'
download_audio(url, filename)




import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Cargar el archivo de audio con librosa
y, sr = librosa.load(filename)

# Extraer el vector de cromas
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Mostrar el cromagrama (opcional, para visualización)
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', cmap='coolwarm')
plt.colorbar()
plt.title('Cromagrama')
plt.tight_layout()
plt.show()





import librosa
import numpy as np

# Definir las tríadas mayores y menores
major_chords = {
    'C': [0, 4, 7],
    'C#': [1, 5, 8],
    'D': [2, 6, 9],
    'D#': [3, 7, 10],
    'E': [4, 8, 11],
    'F': [5, 9, 0],
    'F#': [6, 10, 1],
    'G': [7, 11, 2],
    'G#': [8, 0, 3],
    'A': [9, 1, 4],
    'A#': [10, 2, 5],
    'B': [11, 3, 6]
}

minor_chords = {
    'Cm': [0, 3, 7],
    'C#m': [1, 4, 8],
    'Dm': [2, 5, 9],
    'D#m': [3, 6, 10],
    'Em': [4, 7, 11],
    'Fm': [5, 8, 0],
    'F#m': [6, 9, 1],
    'Gm': [7, 10, 2],
    'G#m': [8, 11, 3],
    'Am': [9, 0, 4],
    'A#m': [10, 1, 5],
    'Bm': [11, 2, 6]
}

# Función para detectar el acorde
def detect_chord(pitches, chord_dict):
    for chord_name, chord_notes in chord_dict.items():
        if set(chord_notes).issubset(set(pitches)):
            return chord_name
    return "Acorde no identificado"

# Cargar el archivo de audio
filename = 'sample_audio.mp3'  # Asegúrate de tener tu archivo aquí
y, sr = librosa.load(filename)

# Extraer cromas
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Sumar las cromas en el tiempo para obtener las notas dominantes
chroma_sum = np.sum(chroma, axis=1)

# Seleccionar las tres notas más dominantes
dominant_pitches = np.argsort(chroma_sum)[-3:]

# Detectar acorde mayor
detected_major_chord = detect_chord(dominant_pitches, major_chords)

# Detectar acorde menor
detected_minor_chord = detect_chord(dominant_pitches, minor_chords)

# Mostrar los resultados
print(f"Acorde mayor detectado: {detected_major_chord}")
print(f"Acorde menor detectado: {detected_minor_chord}")





import librosa
import numpy as np

# Definir tríadas mayores, menores, y séptimos
chords = {
    'C': [0, 4, 7], 'C#m': [1, 4, 8], 'D': [2, 6, 9], 'Dm': [2, 5, 9], 
    'D#': [3, 7, 10], 'D#m': [3, 6, 10], 'E': [4, 8, 11], 'Em': [4, 7, 11],
    'F': [5, 9, 0], 'F#m': [6, 9, 1], 'G': [7, 11, 2], 'Gm': [7, 10, 2],
    'G#': [8, 0, 3], 'A': [9, 1, 4], 'Am': [9, 0, 4], 'A#': [10, 2, 5],
    'B': [11, 3, 6], 'Bm': [11, 2, 6]
}

# Función para detectar acordes
def detect_chord(pitches, chord_dict):
    for chord_name, chord_notes in chord_dict.items():
        # Verificar si las notas detectadas corresponden a un acorde conocido
        if set(chord_notes).issubset(set(pitches)):
            return chord_name
    return "Acorde no identificado"

# Cargar archivo de audio
filename = 'sample_audio.mp3'  # Cambia esto por tu archivo
y, sr = librosa.load(filename)

# Extraer cromas
chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048)

# Sumar las cromas para obtener las notas predominantes en la canción
# chroma_sum = np.sum(chroma, axis=1)

print(len(chroma[0]))

for v in chroma:
    # Filtrar notas más dominantes (por ejemplo, aquellas con más del 30% de la energía máxima)
    threshold = 0.3 * np.max(v)
    dominant_pitches = np.where(v > threshold)[0]

    # Detectar acorde
    detected_chord = detect_chord(dominant_pitches, chords)

    # Mostrar acorde detectado
    # print(f"Acorde detectado: {detected_chord}")





from mingus.containers import Bar
from mingus.midi import fluidsynth

fluidsynth.init("soundfont.sf2")  # Necesitas un archivo soundfont para la reproducción

bar = Bar()
bar.place_notes(["C-4", "E-4", "G-4"], 4)
bar.place_notes(["F-4", "A-4", "C-5"], 4)
fluidsynth.play_Bar(bar)