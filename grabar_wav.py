"""Graba clips de audio en formato .wav reales para poder escucharlos
en la PC -- util para diagnosticar si el microfono capta voz limpia o
ruido/interferencia.

Uso (por REPL):
    >>> import grabar_wav
    >>> grabar_wav.grabar("prueba1.wav", segundos=4)

Luego copialo a la PC:
    mpremote connect COM3 cp :prueba1.wav .

Y reproducelo con cualquier reproductor (Windows Media Player, VLC, etc).
"""

import struct

import audio_i2s
import config


def _cabecera_wav(n_bytes_datos, sample_rate=config.SAMPLE_RATE):
    bits_por_muestra = 16
    canales = 1
    byte_rate = sample_rate * canales * bits_por_muestra // 8
    block_align = canales * bits_por_muestra // 8

    return (
        b"RIFF"
        + struct.pack("<I", 36 + n_bytes_datos)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<HHIIHH", 1, canales, sample_rate, byte_rate, block_align, bits_por_muestra)
        + b"data"
        + struct.pack("<I", n_bytes_datos)
    )


def grabar(nombre_archivo="prueba.wav", segundos=4):
    i2s = audio_i2s.iniciar_i2s()
    lector = audio_i2s.LectorAudio(i2s)

    n_bloques = int(segundos * config.SAMPLE_RATE / config.BLOCK_SAMPLES)
    print("Grabando {} segundos...".format(segundos))

    datos_16bit = bytearray()
    for _ in range(n_bloques):
        muestras = lector.leer_bloque()
        for s in muestras:
            # 24 bits utiles quedan justificados a la izquierda dentro
            # del entero de 32 bits; nos quedamos con los 16 bits mas
            # significativos para un PCM de 16 bits estandar.
            s16 = s >> 16
            if s16 > 32767:
                s16 = 32767
            elif s16 < -32768:
                s16 = -32768
            datos_16bit += struct.pack("<h", s16)

    i2s.deinit()

    with open(nombre_archivo, "wb") as f:
        f.write(_cabecera_wav(len(datos_16bit)))
        f.write(datos_16bit)

    print("Guardado: {} ({} bytes de audio)".format(nombre_archivo, len(datos_16bit)))
    print("Copialo a tu PC con:")
    print("  mpremote connect COM3 cp :{} .".format(nombre_archivo))
