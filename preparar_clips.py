"""Recorta las palabras del vocabulario a la voz y las deja listas para la placa.

Lee grabaciones/vocabulario_hora_fecha/<persona>_<palabra>.wav (2,5 s cada una,
con silencio alrededor) y escribe clips/<persona>_<palabra>.wav: solo la parte
con voz, con 50 ms de margen, mono, 16 kHz, 16 bits. Asi la respuesta suena sin
pausas largas y los clips ocupan poco espacio en la placa.

Uso (desde la carpeta del proyecto):
    python preparar_clips.py felipe
"""

import argparse
import os

import numpy as np
import soundfile as sf

import config
import preparar_dataset_pc as pdp

CARPETA_ORIGEN = os.path.join("grabaciones", "vocabulario_hora_fecha")
CARPETA_SALIDA = "clips"
MARGEN_MS = 120        # margen a cada lado: las consonantes finales (la "ch" de noche) tienen poca energia
FRACCION_PICO = 0.003  # voz = bloques con energia de al menos el 0,3 % de la del bloque mas fuerte (-25 dB)


def recortar(x):
    """Devuelve la parte de la senal entre el primer y el ultimo bloque con voz."""
    B = config.BLOCK_SAMPLES
    bloques = [x[i:i + B] for i in range(0, len(x) - B + 1, B)]
    energias = np.array([pdp.analizar_bloque(b)[0] for b in bloques])
    if len(energias) == 0:
        return x
    # umbral relativo al bloque mas fuerte: el piso de ruido del PC deja colas largas
    # de ruido que el VAD tomaria como voz, y la palabra queda demasiado larga
    umbral = energias.max() * FRACCION_PICO
    con_voz = np.flatnonzero(energias > umbral)
    if con_voz.size == 0:
        return x
    margen = int(MARGEN_MS * config.SAMPLE_RATE / 1000)
    ini = max(0, con_voz[0] * B - margen)
    fin = min(len(x), (con_voz[-1] + 1) * B + margen)
    return x[ini:fin]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("persona", help="nombre de quien grabo, por ejemplo felipe")
    args = ap.parse_args()

    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    prefijo = args.persona + "_"
    archivos = sorted(f for f in os.listdir(CARPETA_ORIGEN) if f.startswith(prefijo) and f.endswith(".wav"))
    if not archivos:
        raise SystemExit("no hay grabaciones de {} en {}".format(args.persona, CARPETA_ORIGEN))

    total = 0
    for nombre in archivos:
        palabra = nombre[len(prefijo):-len(".wav")]
        x = pdp.leer_wav_mono_16k(os.path.join(CARPETA_ORIGEN, nombre))
        recorte = recortar(x).astype(np.int16)
        # hora.py busca clips/<persona>_<palabra>.wav (ver secuencia()): sin el
        # prefijo de persona, la respuesta de hora.py no encuentra el archivo.
        ruta = os.path.join(CARPETA_SALIDA, args.persona + "_" + palabra + ".wav")
        sf.write(ruta, recorte, config.SAMPLE_RATE, subtype="PCM_16")
        total += os.path.getsize(ruta)
        print("  {:<18} {:.2f} s".format(palabra, len(recorte) / config.SAMPLE_RATE))
    print("{} clips en {}/ ({:.0f} KB en total)".format(len(archivos), CARPETA_SALIDA, total / 1024))


if __name__ == "__main__":
    main()
