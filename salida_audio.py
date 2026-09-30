"""Reproduccion de audio por I2S hacia el MAX98357A.

Usa el SEGUNDO periferico I2S de la placa (I2S(1, ...)), separado del
microfono (I2S(0, ...) en audio_i2s.py), asi grabar y reproducir no chocan
entre si.

Reproduce archivos .wav mono de 16 bits -- el mismo formato que genera
grabar_wav.py. Para grabar la frase de confirmacion en tu propia voz:

    >>> import grabar_wav
    >>> grabar_wav.grabar(config.RUTA_CLIP_CONFIRMACION, segundos=2)

Nota: aunque el archivo es de 16 bits, el I2S se abre en modo TX con
bits=32 -- probado en placa: con bits=16 el MAX98357A no emitia sonido
(la señal salia del ESP32 pero el amplificador no la reproducia), y con
bits=32 (muestra de 16 bits desplazada a la izquierda, en los bits altos)
si funciona.
"""

import array
import struct
from machine import I2S, Pin

import config

_TAM_BLOQUE_MUESTRAS = 512  # muestras de 16 bits procesadas por iteracion


def iniciar_i2s_salida(sample_rate):
    return I2S(
        1,
        sck=Pin(config.PIN_AMP_BCLK),
        ws=Pin(config.PIN_AMP_LRC),
        sd=Pin(config.PIN_AMP_DIN),
        mode=I2S.TX,
        bits=32,
        format=I2S.MONO,
        rate=sample_rate,
        ibuf=config.I2S_BUFFER_BYTES,
    )


def _leer_cabecera_wav(f):
    """Lee la cabecera de un WAV PCM mono de 16 bits y devuelve
    (sample_rate, tamano_datos_en_bytes), dejando el archivo posicionado
    justo al inicio de los datos de audio.

    Recorre los bloques ("chunks") del archivo en vez de asumir que el
    audio empieza siempre en el byte 44: varias apps de grabacion (celular,
    editores de audio) insertan bloques extra como "LIST" entre "fmt " y
    "data", y asumir una posicion fija manda metadatos al amplificador en
    vez de audio."""
    cabecera = f.read(12)
    if len(cabecera) < 12 or cabecera[0:4] != b"RIFF" or cabecera[8:12] != b"WAVE":
        raise ValueError("no es un archivo WAV valido")

    sample_rate = canales = bits = tam_datos = None

    while tam_datos is None:
        encabezado_bloque = f.read(8)
        if len(encabezado_bloque) < 8:
            break
        id_bloque = encabezado_bloque[0:4]
        tam_bloque = struct.unpack("<I", encabezado_bloque[4:8])[0]

        if id_bloque == b"fmt ":
            datos_fmt = f.read(tam_bloque)
            canales = struct.unpack_from("<H", datos_fmt, 2)[0]
            sample_rate = struct.unpack_from("<I", datos_fmt, 4)[0]
            bits = struct.unpack_from("<H", datos_fmt, 14)[0]
            if tam_bloque % 2:
                f.seek(1, 1)
        elif id_bloque == b"data":
            tam_datos = tam_bloque
            # no se salta nada: el puntero ya queda al inicio del audio
        else:
            f.seek(tam_bloque + (tam_bloque % 2), 1)

    if sample_rate is None or tam_datos is None:
        raise ValueError("WAV sin bloque fmt o data")
    if canales != 1 or bits != 16:
        raise ValueError("se espera un WAV mono de 16 bits")
    return sample_rate, tam_datos


def reproducir(nombre_archivo, i2s=None):
    """Reproduce un .wav mono de 16 bits por el amplificador. Si no se
    pasa `i2s`, crea y libera uno propio; para reproducir varios clips
    seguidos sin el "clic" de reabrir el periferico cada vez, crealo una
    vez con iniciar_i2s_salida() y pasalo."""
    with open(nombre_archivo, "rb") as f:
        sample_rate, tam_datos = _leer_cabecera_wav(f)
        propio = i2s is None
        if propio:
            i2s = iniciar_i2s_salida(sample_rate)
        try:
            entrada = bytearray(_TAM_BLOQUE_MUESTRAS * 2)
            salida = bytearray(_TAM_BLOQUE_MUESTRAS * 4)
            mv_entrada = memoryview(entrada)
            mv_salida = memoryview(salida)
            restante = tam_datos
            while restante > 0:
                pedir = min(len(entrada), restante)
                n_bytes = f.readinto(mv_entrada[:pedir])
                if not n_bytes:
                    break
                n_muestras = n_bytes // 2
                muestras = array.array("h", entrada[: n_muestras * 2])
                for j in range(n_muestras):
                    struct.pack_into("<i", salida, j * 4, muestras[j] << 16)
                i2s.write(mv_salida[: n_muestras * 4])
                restante -= n_bytes
        finally:
            if propio:
                i2s.deinit()
