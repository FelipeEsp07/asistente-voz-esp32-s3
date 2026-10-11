"""Graba en lote, con el microfono de la PC, las frases de prueba.

Por cada toma: pulsas Enter, se graba la frase durante SEGUNDOS y se guarda.
Las tomas se guardan
en grabaciones/lote/ con el nombre felipe_<frase>_<n>.wav (WAV mono, 16 bits,
16 kHz). Si una toma sale saturada o casi en silencio, se repite sin contarla.
Nunca sobrescribe: si ya existe el numero, lo salta.

Uso:
    python grabar_lote.py
    python grabar_lote.py --tomas 10          (prueba corta)
"""

import argparse
import os

import sounddevice as sd

import grabar_wav as gw

# (nombre de archivo, texto que se muestra al grabar)
FRASES = [
    ("fecha", "qué fecha es hoy"),
    ("activar_motobomba", "activar motobomba"),
    ("activar_bombillo", "activar bombillo"),
]
CARPETA = os.path.join("grabaciones", "lote")
SEGUNDOS = 2.5


def _microfono_intel():
    """Indice del microfono Intel (MME). Los indices cambian entre sesiones, por eso se busca por nombre.
    Devuelve None si no existe y se usa el predeterminado."""
    for indice, dispositivo in enumerate(sd.query_devices()):
        if dispositivo["max_input_channels"] >= 1 and "Intel" in dispositivo["name"]:
            if sd.query_hostapis(dispositivo["hostapi"])["name"] == "MME":
                return indice
    return None


def _grabar_una_toma(dispositivo):
    """Graba una toma; repite si el nivel no sirve."""
    while True:
        print("  grabando...", flush=True)
        audio = gw._grabar_toma(SEGUNDOS, dispositivo)
        problema = gw._problema_de_nivel(audio)
        if problema is None:
            return audio
        print("  [AVISO] {}. Se repite la toma.".format(problema))


def main():
    parser = argparse.ArgumentParser(description="Graba lotes de frases con el microfono de la PC.")
    parser.add_argument("--tomas", type=int, default=50, help="tomas por frase (por defecto 50)")
    args = parser.parse_args()

    os.makedirs(CARPETA, exist_ok=True)
    dispositivo = _microfono_intel()
    print("Microfono: {}".format(sd.query_devices(dispositivo)["name"] if dispositivo is not None else "predeterminado"))
    try:
        for frase, texto in FRASES:
            print("\n=== {} ({} tomas) ===".format(texto, args.tomas))
            for n in range(1, args.tomas + 1):
                ruta = os.path.join(CARPETA, "felipe_{}_{}.wav".format(frase, n))
                if os.path.exists(ruta):
                    print("  {} ya existe, se omite".format(os.path.basename(ruta)))
                    continue
                input("[{:02d}/{}] pulsa Enter y di: \"{}\"".format(n, args.tomas, texto))
                audio = _grabar_una_toma(dispositivo)
                gw.sf.write(ruta, audio, gw.FS, subtype="PCM_16")
                print("  guardado {}".format(os.path.basename(ruta)))
    except KeyboardInterrupt:
        print("\nDetenido. Las tomas ya guardadas se conservan.")
        return
    print("\nListo. Tomas en {}/".format(CARPETA))


if __name__ == "__main__":
    main()
