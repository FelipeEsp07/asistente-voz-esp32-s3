"""Diagnostico: imprime la energia cruda de cada bloque de audio, sin
pasar por el VAD. Sirve para calibrar UMBRAL_RUIDO_INICIAL y
FACTOR_ACTIVACION viendo numeros reales del ambiente/microfono.

Uso (por REPL):
    >>> import diagnostico_vad
    >>> diagnostico_vad.ejecutar()

Corre ~10 segundos: los primeros segundos quedate en silencio, despues
di "activar asistente" un par de veces.
"""

import time

import audio_i2s
import config


def ejecutar(segundos=10):
    i2s = audio_i2s.iniciar_i2s()
    lector = audio_i2s.LectorAudio(i2s)

    n_bloques = int(segundos * config.SAMPLE_RATE / config.BLOCK_SAMPLES)
    print("Grabando {} bloques (~{}s). Quedate en silencio al inicio,".format(n_bloques, segundos))
    print("luego di 'activar asistente' un par de veces.")
    time.sleep(1)

    bloques_por_segundo = max(1, config.SAMPLE_RATE // config.BLOCK_SAMPLES)
    energias = []
    for i in range(n_bloques):
        muestras = lector.leer_bloque()
        energia = lector.analizar(muestras)[0]
        energias.append(energia)

    i2s.deinit()

    print("\n--- Perfil por segundo (min / promedio / max) ---")
    for seg in range(0, n_bloques, bloques_por_segundo):
        trozo = energias[seg: seg + bloques_por_segundo]
        if not trozo:
            continue
        t = seg // bloques_por_segundo
        print("t={}s  min={}  avg={}  max={}".format(
            t, min(trozo), sum(trozo) // len(trozo), max(trozo)
        ))

    print("\n--- Resumen ---")
    print("Energia minima:", min(energias))
    print("Energia maxima:", max(energias))
    print("Umbral actual (piso_ruido x factor):", config.UMBRAL_RUIDO_INICIAL * config.FACTOR_ACTIVACION)
