"""Graba tomas de voz con el microfono del circuito (INMP441) para el entrenamiento.

Se usa en el REPL de la placa, sin el asistente en marcha:

    >>> import grabar_placa
    >>> grabar_placa.tomas("fecha_y_hora", 30)

Por cada toma: pulsas Enter, se graba durante SEGUNDOS y se guarda en la placa
como <frase>_<n>.wav (WAV mono, 16 bits, 16 kHz). Si la toma sale saturada o
casi en silencio, se avisa y se repite sin contarla. Nunca sobrescribe: si el
numero ya existe, continua con el siguiente.

Para pasar las tomas a la PC:
    python copiar_grabaciones_placa.py COM5 fecha_y_hora felipe
"""

import os
import struct

import audio_i2s
import config

SEGUNDOS = 2
PICO_SATURADO = 32000   # de 32767: si llega aqui, la voz se corta
PICO_BAJO = 1000        # de 32767: casi silencio


def _cabecera_wav(n_bytes_datos, sample_rate):
    return (
        b"RIFF" + struct.pack("<I", 36 + n_bytes_datos) + b"WAVE"
        + b"fmt " + struct.pack("<I", 16)
        + struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data" + struct.pack("<I", n_bytes_datos)
    )


def _siguiente_numero(frase):
    prefijo = frase + "_"
    numeros = [0]
    for nombre in os.listdir():
        if nombre.startswith(prefijo) and nombre.endswith(".wav"):
            try:
                numeros.append(int(nombre[len(prefijo):-4]))
            except ValueError:
                pass
    return max(numeros) + 1


def _grabar_toma(lector, segundos):
    """Devuelve los datos de audio en 16 bits (bytearray) y el pico absoluto."""
    n_bloques = int(segundos * config.SAMPLE_RATE / config.BLOCK_SAMPLES)
    datos = bytearray(n_bloques * config.BLOCK_SAMPLES * 2)
    pico = 0
    pos = 0
    for _ in range(n_bloques):
        muestras = lector.leer_bloque()
        for s in muestras:
            s16 = s >> 16  # el INMP441 entrega 24 bits en un entero de 32
            if s16 > 32767:
                s16 = 32767
            elif s16 < -32768:
                s16 = -32768
            struct.pack_into("<h", datos, pos, s16)
            pos += 2
            if abs(s16) > pico:
                pico = abs(s16)
    return datos, pico


def tomas(frase, cantidad, instruccion=None):
    """Graba `cantidad` tomas de `frase` con el microfono de la placa.

    instruccion: que se muestra antes de cada toma. Puede ser:
    - None: dice la frase (por defecto).
    - un texto: el mismo para todas las tomas (para ruido sin hablar, por ejemplo).
    - una lista de textos: se muestra uno distinto en cada toma, en orden, y se
      repite la lista si `cantidad` es mayor (util para negativos variados, por
      ejemplo frases parecidas a la de activacion pero que no son la frase)."""
    i2s = audio_i2s.iniciar_i2s()
    lector = audio_i2s.LectorAudio(i2s, descartar_arranque=False)
    numero = _siguiente_numero(frase)
    hechas = 0
    if instruccion is None:
        instruccion = "di \"{}\"".format(frase.replace("_", " "))
    lista = instruccion if isinstance(instruccion, list) else None
    print("Grabando '{}': {} tomas, {} s cada una".format(frase, cantidad, SEGUNDOS))
    try:
        while hechas < cantidad:
            texto = lista[hechas % len(lista)] if lista else instruccion
            input("[{}/{}] pulsa Enter y {}".format(hechas + 1, cantidad, texto))
            print("  grabando...")
            datos, pico = _grabar_toma(lector, SEGUNDOS)
            if pico >= PICO_SATURADO:
                print("  [AVISO] toma saturada (la voz se corta). Se repite.")
                continue
            if pico < PICO_BAJO:
                print("  [AVISO] toma muy baja (casi silencio). Se repite.")
                continue
            nombre = "{}_{:02d}.wav".format(frase, numero)
            with open(nombre, "wb") as f:
                f.write(_cabecera_wav(len(datos), config.SAMPLE_RATE))
                f.write(datos)
            print("  guardado {}".format(nombre))
            numero += 1
            hechas += 1
    except KeyboardInterrupt:
        print("\nDetenido. Las tomas guardadas se conservan.")
    finally:
        i2s.deinit()
    print("Listo: {} tomas de '{}' en la placa".format(hechas, frase))
