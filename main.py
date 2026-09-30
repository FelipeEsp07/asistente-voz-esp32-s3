"""Asistente de voz local para ESP32-S3 + INMP441 (MicroPython).

Escucha continuamente y, cuando el modelo reconoce la frase "activar
asistente", confirma por consola, enciende un LED y responde con una
frase corta grabada en la propia voz del usuario (config.RUTA_CLIP_CONFIRMACION,
ver salida_audio.py).

Requiere modelo_pesos.py, generado en la PC con entrenar_modelo.py a partir
de las muestras grabadas con recolectar_datos.py.
"""

import time

import audio_i2s
import config
import feature_extractor
import inferencia
import led
import modelo_pesos as m
import salida_audio
import vad


def ejecutar():
    print("Iniciando asistente de voz local...")

    i2s = audio_i2s.iniciar_i2s()
    lector = audio_i2s.LectorAudio(i2s)
    detector = vad.DetectorVoz()

    print("Listo. Esperando 'activar asistente'...")

    while True:
        muestras = lector.leer_bloque()
        energia, carga = lector.analizar(muestras)
        segmento = detector.procesar_bloque(energia, carga)

        if segmento is None:
            continue

        caracteristicas = feature_extractor.extraer(segmento)
        probabilidad = inferencia.predecir_probabilidad(caracteristicas, m)

        if probabilidad >= config.UMBRAL_CONFIANZA:
            print("[COMANDO RECONOCIDO] 'activar asistente' (confianza {:.2f}).".format(probabilidad))
            led.encender()

            try:
                salida_audio.reproducir(config.RUTA_CLIP_CONFIRMACION)
            except OSError:
                print("[AVISO] No se encontro {}.".format(config.RUTA_CLIP_CONFIRMACION))
            except ValueError as error:
                print("[AVISO] {} no es un WAV valido: {}".format(config.RUTA_CLIP_CONFIRMACION, error))

            # El microfono siguio acumulando audio mientras sonaba el
            # parlante (el eco de la propia respuesta); se descarta antes
            # de volver a escuchar para no confundirlo con una orden nueva.
            lector.vaciar_buffer()

            time.sleep(config.SEGUNDOS_LED_ENCENDIDO)
            led.apagar()
            print("Esperando 'activar asistente'...")
        else:
            print("[DESCARTADO] sonido no coincide (confianza {:.2f})".format(probabilidad))


if __name__ == "__main__":
    ejecutar()
