"""Convierte un audio (de cualquier formato/canales/bits que soporte tu
sistema) al WAV mono de 16 bits que necesita salida_audio.py en la placa.

Requiere ffmpeg instalado en el sistema (no se instala nada en la placa,
este script corre en la PC).

Uso:
    python convertir_audio.py audios/1.wav confirmacion.wav
    python convertir_audio.py audios/1.wav confirmacion.wav --rate 16000
"""

import argparse
import subprocess
import sys


def convertir(origen, destino, rate=None):
    cmd = ["ffmpeg", "-y", "-i", origen, "-ac", "1", "-sample_fmt", "s16"]
    if rate:
        cmd += ["-ar", str(rate)]
    cmd += [destino]
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode != 0:
        print(resultado.stderr[-2000:])
        sys.exit(1)
    print("Convertido:", destino)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("origen")
    ap.add_argument("destino")
    ap.add_argument("--rate", type=int, default=None, help="frecuencia de muestreo (por defecto, la del original)")
    args = ap.parse_args()
    convertir(args.origen, args.destino, args.rate)
