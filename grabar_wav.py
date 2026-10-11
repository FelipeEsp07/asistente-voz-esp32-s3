"""Graba tomas de voz con el microfono de la PC para entrenar el asistente.

Dos modos:

1. Tomas para entrenar (grupo + cantidad). Cada toma se guarda como
   grabaciones/<grupo>/<persona>_<grupo>_<numero>.wav. Nunca sobrescribe:
   la numeracion continua desde la ultima toma existente.

2. Vocabulario (--vocabulario). Una toma por palabra, con la palabra escrita en
   pantalla: grabaciones/vocabulario_hora_fecha/<persona>_<palabra>.wav. Las
   palabras que ya existen se omiten, asi que puedes parar y seguir despues.

Todas las tomas son WAV mono de 16 bits a 16 kHz, el formato que usa la placa.

Uso:
    python grabar_wav.py --listar-dispositivos
    python grabar_wav.py felipe hola_pan 10
    python grabar_wav.py felipe negativos 30 --segundos 2.5 --dispositivo 3
    python grabar_wav.py felipe --vocabulario

Por cada toma: pulsa Enter, di la frase y espera a que termine la grabacion.
Si la toma sale saturada o casi en silencio se avisa y se repite.
"""

import argparse
import glob
import os
import re
import unicodedata

import numpy as np
import sounddevice as sd
import soundfile as sf

FS = 16000
CARPETA_BASE = "grabaciones"
PICO_SATURADO = 0.99
PICO_BAJO = 0.05

GRUPO_VOCABULARIO = "vocabulario_hora_fecha"
# Palabras de la seccion 6 de GUIA_GRABACIONES.md. Con "una" se cubren las horas 1 y 21.
VOCABULARIO = (
    ["cero", "uno", "una", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
    + ["diez", "once", "doce", "trece", "catorce", "quince", "dieciseis", "diecisiete", "dieciocho", "diecinueve"]
    + ["veinte", "veintiuno", "veintidos", "veintitres", "veinticuatro", "veinticinco",
       "veintiseis", "veintisiete", "veintiocho", "veintinueve"]
    + ["treinta", "cuarenta", "cincuenta", "y"]
    + ["son_las", "es_la", "menos", "en_punto", "cuarto", "media", "hoy_es", "hoy_estamos_a", "de", "del", "dos_mil"]
    + ["de_la_manana", "de_la_tarde", "de_la_noche"]
    + ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
    + ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
       "septiembre", "octubre", "noviembre", "diciembre"]
)


def _normalizar(texto):
    sin_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", sin_acentos.lower()).strip("_")


def _siguiente_numero(carpeta, persona, grupo):
    patron = os.path.join(carpeta, "{}_{}_*.wav".format(persona, grupo))
    numeros = []
    for ruta in glob.glob(patron):
        coincidencia = re.search(r"_(\d+)\.wav$", ruta)
        if coincidencia:
            numeros.append(int(coincidencia.group(1)))
    return max(numeros, default=0) + 1


def _grabar_toma(segundos, dispositivo):
    audio = sd.rec(int(segundos * FS), samplerate=FS, channels=1, dtype="float32", device=dispositivo)
    sd.wait()
    return audio[:, 0]


def _problema_de_nivel(audio):
    pico = float(np.max(np.abs(audio)))
    if pico >= PICO_SATURADO:
        return "toma saturada (la voz se corta): aleja un poco el microfono o baja la ganancia"
    if pico < PICO_BAJO:
        return "toma muy baja (casi silencio): acerca el microfono o sube la ganancia"
    return None


def _grabar_hasta_tener_toma_buena(segundos, dispositivo):
    while True:
        audio = _grabar_toma(segundos, dispositivo)
        problema = _problema_de_nivel(audio)
        if problema is None:
            return audio
        print("  [AVISO] {}. Repite la toma.".format(problema))
        input("  pulsa Enter y repite...")


def _modo_entrenamiento(args, persona, grupo):
    carpeta = os.path.join(args.carpeta, grupo)
    os.makedirs(carpeta, exist_ok=True)

    numero = _siguiente_numero(carpeta, persona, grupo)
    hechas = 0
    print("Grabando '{}' para {} en {}/".format(grupo, persona, carpeta))
    while hechas < args.cantidad:
        input("Toma {:02d} ({}/{}): pulsa Enter y di la frase...".format(numero, hechas + 1, args.cantidad))
        audio = _grabar_hasta_tener_toma_buena(args.segundos, args.dispositivo)
        ruta = os.path.join(carpeta, "{}_{}_{:02d}.wav".format(persona, grupo, numero))
        sf.write(ruta, audio, FS, subtype="PCM_16")
        print("  guardado {}".format(ruta))
        numero += 1
        hechas += 1
    print("Listo: {} tomas en {}/".format(hechas, carpeta))


def _modo_vocabulario(args, persona):
    carpeta = os.path.join(args.carpeta, GRUPO_VOCABULARIO)
    os.makedirs(carpeta, exist_ok=True)

    pendientes = [p for p in VOCABULARIO if not os.path.exists(os.path.join(carpeta, "{}_{}.wav".format(persona, p)))]
    print("Vocabulario: {} palabras; faltan {} en {}/".format(len(VOCABULARIO), len(pendientes), carpeta))
    for indice, palabra in enumerate(pendientes, 1):
        texto = palabra.replace("_", " ")
        input("[{}/{}] Di: \"{}\" y pulsa Enter para grabar...".format(indice, len(pendientes), texto))
        audio = _grabar_hasta_tener_toma_buena(args.segundos, args.dispositivo)
        ruta = os.path.join(carpeta, "{}_{}.wav".format(persona, palabra))
        sf.write(ruta, audio, FS, subtype="PCM_16")
        print("  guardado {}".format(ruta))
    print("Listo: vocabulario de {} en {}/".format(persona, carpeta))


GRUPO_PREGUNTAS = "preguntas_hora_fecha"
TOMAS_POR_PREGUNTA = 15
# (nombre de archivo, texto para mostrar). Son las frases que dice la persona al asistente.
PREGUNTAS = [
    ("fecha_y_hora", "fecha y hora"),
]


def _modo_preguntas(args, persona):
    carpeta = os.path.join(args.carpeta, GRUPO_PREGUNTAS)
    os.makedirs(carpeta, exist_ok=True)
    frases = [(s, t) for s, t in PREGUNTAS if args.frase is None or s == args.frase]
    if not frases:
        raise SystemExit("frase desconocida: {}. Opciones: {}".format(args.frase, ", ".join(s for s, _ in PREGUNTAS)))
    tomas = args.tomas or TOMAS_POR_PREGUNTA
    print("Preguntas: {} frase(s), {} tomas cada una, en {}/".format(len(frases), tomas, carpeta))

    for slug, texto in frases:
        patron = re.compile(r"{}_{}_(\d+)\.wav".format(persona, slug))
        numeros = [int(m.group(1)) for m in map(patron.fullmatch, os.listdir(carpeta)) if m]
        siguiente = max(numeros, default=0) + 1
        for numero in range(siguiente, tomas + 1):
            input("[{}] toma {}/{}: di \"{}\" y pulsa Enter para grabar...".format(slug, numero, tomas, texto))
            audio = _grabar_hasta_tener_toma_buena(args.segundos, args.dispositivo)
            ruta = os.path.join(carpeta, "{}_{}_{:02d}.wav".format(persona, slug, numero))
            sf.write(ruta, audio, FS, subtype="PCM_16")
            print("  guardado {}".format(ruta))
    print("Listo: preguntas de {} en {}/".format(persona, carpeta))


def main():
    parser = argparse.ArgumentParser(description="Graba tomas de voz con el microfono de la PC.")
    parser.add_argument("persona", nargs="?", help="nombre de quien habla, por ejemplo felipe")
    parser.add_argument("grupo", nargs="?", help="grupo de la guia, por ejemplo hola_pan")
    parser.add_argument("cantidad", nargs="?", type=int, help="tomas a grabar")
    parser.add_argument("--vocabulario", action="store_true", help="graba una toma por palabra de hora y fecha")
    parser.add_argument("--preguntas", action="store_true", help="graba las frases de hora y fecha que dice la persona")
    parser.add_argument("--frase", default=None, help="con --preguntas: solo esta frase (ej. fecha_de_hoy)")
    parser.add_argument("--tomas", type=int, default=None, help="con --preguntas: tomas totales por frase (por defecto 15)")
    parser.add_argument("--segundos", type=float, default=2.5, help="duracion de cada toma (por defecto 2.5)")
    parser.add_argument("--dispositivo", type=int, default=None, help="indice del microfono (ver --listar-dispositivos)")
    parser.add_argument("--carpeta", default=CARPETA_BASE, help="carpeta base (por defecto grabaciones)")
    parser.add_argument("--listar-dispositivos", action="store_true", help="muestra los microfonos disponibles y sale")
    args = parser.parse_args()

    if args.listar_dispositivos:
        print(sd.query_devices())
        return
    if not args.persona:
        parser.error("indica la persona")
    persona = _normalizar(args.persona)

    try:
        if args.vocabulario:
            _modo_vocabulario(args, persona)
        elif args.preguntas:
            _modo_preguntas(args, persona)
        else:
            if not (args.grupo and args.cantidad):
                parser.error("indica grupo y cantidad, o usa --vocabulario")
            _modo_entrenamiento(args, persona, _normalizar(args.grupo))
    except KeyboardInterrupt:
        print("\nDetenido. Las tomas ya guardadas se conservan.")


if __name__ == "__main__":
    main()
