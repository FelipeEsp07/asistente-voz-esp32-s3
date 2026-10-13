"""Convierte grabaciones WAV hechas en la PC al formato de dataset.jsonl.

Las muestras de la PC deben procesarse EXACTAMENTE como las de la placa,
si no el modelo entrena con una representacion distinta a la que usa en
uso real. Por eso este script reproduce:
  - mono, 16 kHz, enteros de 16 bits (con filtro anti-aliasing al remuestrear);
  - bloques de config.BLOCK_SAMPLES muestras, sin solapamiento;
  - por bloque: se quita el offset DC y se calculan rms16, hf y zcr igual que
    audio_i2s.LectorAudio.analizar;
  - segmentacion con el mismo vad.DetectorVoz de la placa.

Como el nivel de la grabacion de PC no es el del INMP441, el piso de ruido
del VAD se estima de cada archivo (percentil 10 de la energia) en vez de usar
el umbral absoluto de config.UMBRAL_RUIDO_INICIAL.

Uso (desde la carpeta del proyecto):
    python preparar_dataset_pc.py grabaciones_hola_pan --label 1 --hablante felipe
    python preparar_dataset_pc.py grabaciones_otras --label 0
    python preparar_dataset_pc.py grabaciones_comandos --label 11  (palabra "comandos", ver CLASES)
"""

import argparse
import json
import math
import os
import re
import wave
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

import config
import espectro
import tono
import vad

ESCALA_ENERGIA = 2 ** 32  # misma escala que audio_i2s.LectorAudio.analizar


def leer_wav_mono_16k(ruta):
    """Devuelve la señal como enteros de 16 bits, mono, a 16 kHz."""
    with wave.open(ruta, "rb") as w:
        canales = w.getnchannels()
        ancho = w.getsampwidth()
        tasa = w.getframerate()
        crudo = w.readframes(w.getnframes())

    if ancho == 2:
        x = np.frombuffer(crudo, dtype="<i2").astype(float)
    elif ancho == 3:
        b = np.frombuffer(crudo, dtype=np.uint8).reshape(-1, 3).astype(np.int64)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 2 ** 23, v - 2 ** 24, v)
        x = v.astype(float) / 256
    elif ancho == 4:
        x = np.frombuffer(crudo, dtype="<i4").astype(float) / 65536
    else:
        raise ValueError("ancho de muestra no soportado: {} bytes".format(ancho))

    if canales > 1:
        x = x.reshape(-1, canales).mean(axis=1)

    if tasa != config.SAMPLE_RATE:
        fr = Fraction(config.SAMPLE_RATE, tasa).limit_denominator(1000)
        x = resample_poly(x, fr.numerator, fr.denominator)

    return np.clip(np.round(x), -32768, 32767).astype(np.int64)


def analizar_bloque(bloque):
    """Mismo calculo que audio_i2s.LectorAudio.analizar, sobre enteros de 16 bits."""
    n = len(bloque)
    dc = int(bloque.sum() // n)
    s = bloque - dc
    suma_e = int((s * s).sum())
    d = np.diff(s, prepend=s[0])
    suma_d = int((d * d).sum())
    cruces = int(np.count_nonzero((s[1:] ^ s[:-1]) < 0))

    energia = suma_e * ESCALA_ENERGIA / n
    rms16 = round(math.sqrt(suma_e / n), 1)
    hf = round(suma_d / suma_e, 3) if suma_e > 0 else 0.0
    zcr = round(cruces / (n - 1), 3)
    return energia, (rms16, hf, zcr)


def segmentos_de(x):
    """Devuelve una lista de (carga_por_bloque, bloque_inicial, bloque_final_exclusivo)
    por cada frase detectada. Los indices de bloque sirven para recortar el audio
    crudo del segmento elegido (para el tono, que necesita la señal, no los bloques)."""
    B = config.BLOCK_SAMPLES
    bloques_x = [x[i:i + B] for i in range(0, len(x) - B + 1, B)]
    analisis = [analizar_bloque(b) for b in bloques_x]
    energias = sorted(e for e, _ in analisis)
    piso = energias[len(energias) // 10] or 1

    detector = vad.DetectorVoz()
    detector.piso_ruido = piso
    segmentos = []
    for idx, (energia, carga) in enumerate(analisis):
        seg = detector.procesar_bloque(energia, carga)
        if seg is not None:
            fin = idx + 1
            segmentos.append((seg, fin - len(seg), fin))
    cola = detector.cerrar()  # la frase puede terminar al final del archivo, sin silencio posterior
    if cola is not None:
        fin = len(analisis)
        segmentos.append((cola, fin - len(cola), fin))
    return segmentos


def _hablante_de_archivo(nombre):
    """'laura_hola_bender_01.wav' -> 'laura'. Todas las grabaciones del proyecto
    nombran el archivo empezando por la persona que habla."""
    return nombre.split("_", 1)[0]


def procesar_archivo(ruta, etiqueta, hablante, distancia):
    x = leer_wav_mono_16k(ruta)
    segmentos = segmentos_de(x)
    if not segmentos:
        print("  [OMITIDO] {}: no se detecto ninguna frase".format(os.path.basename(ruta)))
        return None
    if len(segmentos) > 1:
        print("  [AVISO] {}: {} segmentos, se usa el mas largo".format(os.path.basename(ruta), len(segmentos)))
    segmento, ini, fin = max(segmentos, key=lambda s: len(s[0]))
    B = config.BLOCK_SAMPLES
    x_segmento = x[ini * B:fin * B]  # muestras crudas de la frase, no del archivo completo
    f0 = tono.f0_mediana(x_segmento)
    centroide = espectro.centroide_frase(x_segmento)
    return {
        "bloques": [list(b) for b in segmento],
        "label": etiqueta,
        "distancia": distancia,
        "hablante": hablante or _hablante_de_archivo(os.path.basename(ruta)),
        "fuente": "pc",
        "archivo": os.path.basename(ruta),
        "f0": round(float(f0), 1),
        "centroide": round(float(centroide), 4),
    }


# Clases del modelo (mismo orden que FINE_CLASES en entrenar_modelo.py). "comandos"
# (la palabra sola, para pasar al estado ESCUCHA_ACTUADOR) se agrego al final,
# en el indice 11, para no reetiquetar las muestras ya guardadas en dataset.jsonl.
CLASES = ("otro", "activacion", "fecha_hora", "activar_motobomba", "activar_bombillo", "apagar_bombillo",
          "apagar_motobomba", "activar_secador", "apagar_secador", "activar_maquinilla", "apagar_maquinilla",
          "comandos")
# Etiqueta de cada frase de preguntas y comandos, a partir del nombre de archivo.
ETIQUETA_PREGUNTA = {
    "fecha_y_hora": 2, "fecha": 2,  # "fecha y hora" y "que fecha es hoy" disparan la misma respuesta
    "activar_motobomba": 3, "activar_bombillo": 4, "apagar_bombillo": 5, "apagar_motobomba": 6,
    "activar_secador": 7, "apagar_secador": 8, "activar_maquinilla": 9, "apagar_maquinilla": 10,
}


def _etiqueta_de_pregunta(nombre, hablante):
    """'felipe_que_hora_es_03.wav' -> 2. Devuelve None si el nombre no es de una pregunta."""
    coincidencia = re.fullmatch(r"(?:{}_)?(.+)_\d+\.wav".format(re.escape(hablante)), nombre) if hablante else None
    if coincidencia is None:
        coincidencia = re.fullmatch(r"(.+)_\d+\.wav", nombre)
    return ETIQUETA_PREGUNTA.get(coincidencia.group(1)) if coincidencia else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta", help="carpeta con los WAV de la PC")
    ap.add_argument("--label", type=int, choices=range(len(CLASES)), default=None,
                    help="etiqueta fija para toda la carpeta: 0 otro, 1 activacion")
    ap.add_argument("--preguntas", action="store_true",
                    help="la etiqueta sale del nombre de cada archivo (hora, fecha, fecha_y_hora)")
    ap.add_argument("--hablante", default=None, help="nombre de quien habla (ej. felipe)")
    ap.add_argument("--distancia", default="pc", help="etiqueta de distancia (por defecto 'pc')")
    ap.add_argument("--dataset", default=config.RUTA_DATASET, help="archivo .jsonl donde agregar")
    args = ap.parse_args()
    if args.preguntas == (args.label is not None):
        ap.error("usa --label o --preguntas (exactamente uno)")

    archivos = sorted(f for f in os.listdir(args.carpeta) if f.lower().endswith(".wav"))
    if not archivos:
        raise SystemExit("no hay archivos .wav en {}".format(args.carpeta))

    agregadas = 0
    with open(args.dataset, "a", encoding="utf-8") as salida:
        for nombre in archivos:
            if args.preguntas:
                etiqueta = _etiqueta_de_pregunta(nombre, args.hablante)
                if etiqueta is None:
                    print("  [OMITIDO] {}: el nombre no corresponde a una pregunta".format(nombre))
                    continue
            else:
                etiqueta = args.label
            muestra = procesar_archivo(os.path.join(args.carpeta, nombre), etiqueta, args.hablante, args.distancia)
            if muestra is None:
                continue
            salida.write(json.dumps(muestra) + "\n")
            agregadas += 1
            dur = len(muestra["bloques"]) * config.BLOCK_SAMPLES / config.SAMPLE_RATE
            print("  [OK] {}: {:.2f} s".format(nombre, dur))

    print("Agregadas {} de {} grabaciones a {}".format(agregadas, len(archivos), args.dataset))


if __name__ == "__main__":
    main()
