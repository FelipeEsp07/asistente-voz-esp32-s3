"""Asistente de voz local para ESP32-S3 + INMP441 + MAX98357A (MicroPython).

Todo corre en el nucleo principal: captura de audio, VAD, deteccion de la
frase de activacion y de comandos, control de actuadores, reproduccion de
respuestas y watchdog. La reproduccion alimenta el watchdog entre bloques,
asi que un clip largo no lo dispara.

Flujo: frase de activacion -> confirmacion -> escucha de un comando ->
accion (encender/apagar un actuador) -> confirmacion -> espera.

Estados:
- ESPERA: busca la frase de activacion.
- ESCUCHA: espera un comando durante config.SEGUNDOS_ESCUCHA.

Requiere modelo_pesos.py generado con entrenar_modelo.py.
"""

import time

from machine import RTC, WDT, Pin

import actuadores
import audio_i2s
import comandos
import config
import feature_extractor
import hora
import inferencia
import led
import modelo_pesos as m
import salida_audio
import vad

ESPERA = "espera"
ESCUCHA = "escucha"


def _vencido(instante_ms):
    return instante_ms is not None and time.ticks_diff(time.ticks_ms(), instante_ms) >= 0


CLASES = m.CLASES
CLASE_ACTIVACION = CLASES.index("activacion")
PREGUNTAS = ("fecha_hora",)
ANIO_RELOJ_VALIDO = 2026  # si el reloj esta antes, no se fijo: no se responde la hora
MS_SORDO_TRAS_RESPUESTA = 800  # ms sin escuchar tras cada sonido del asistente (su eco por el parlante)


def _hablar(ruta, lector, perro):
    """Reproduce un clip y descarta el eco del microfono al terminar."""
    try:
        salida_audio.reproducir(ruta, alimentar=perro.feed if perro is not None else None)
    except OSError:
        print("[AVISO] No se encontro {}.".format(ruta))
    except ValueError as error:
        print("[AVISO] {} no es un WAV valido: {}".format(ruta, error))
    lector.vaciar_buffer()


def _mejor_pregunta(probs):
    """Nombre de la pregunta de hora/fecha si es la clase mas probable y supera
    el umbral; None en otro caso."""
    k = max(range(len(probs)), key=lambda i: probs[i])
    if CLASES[k] in PREGUNTAS and probs[k] >= config.UMBRAL_PREGUNTA:
        return CLASES[k]
    return None


def _mejor_comando(probs):
    """Etiqueta del comando de activacion si es la clase mas probable y supera
    el umbral de preguntas; None en otro caso."""
    k = max(range(len(probs)), key=lambda i: probs[i])
    if CLASES[k] in config.COMANDOS and probs[k] >= config.UMBRAL_PREGUNTA:
        return CLASES[k]
    return None


def _responder(tipo, lector, perro):
    """Dice la hora, la fecha o ambas uniendo clips de clips/."""
    ahora = RTC().datetime()
    if ahora[0] < ANIO_RELOJ_VALIDO:
        print("[AVISO] El reloj no esta fijado ({}). Fijalo desde la PC.".format(ahora[0]))
        return
    rutas = hora.secuencia(tipo, ahora)
    i2s_salida = salida_audio.iniciar_i2s_salida(config.SAMPLE_RATE)
    alimentar = perro.feed if perro is not None else None
    try:
        for ruta in rutas:
            salida_audio.reproducir(ruta, i2s=i2s_salida, alimentar=alimentar)
    except OSError:
        print("[AVISO] Falta un clip de la respuesta. Revisa la carpeta clips/.")
    finally:
        i2s_salida.deinit()
        lector.vaciar_buffer()


def ejecutar(vigilancia=True):
    print("Iniciando asistente de voz local...")

    i2s = audio_i2s.iniciar_i2s()
    lector = audio_i2s.LectorAudio(i2s)
    detector = vad.DetectorVoz()
    act = actuadores.Actuadores(config.ACTUADORES, lambda n: Pin(n, Pin.OUT, value=actuadores.NIVEL_APAGADO), time.ticks_ms)
    perro = WDT(timeout=config.SEGUNDOS_VIGILANCIA * 1000) if vigilancia else None

    estado = ESPERA
    escucha_hasta = None
    led_apagar_en = None
    sordo_hasta = None  # ventana sin escuchar tras cada respuesta: evita que su eco se tome por voz

    print("Listo. Esperando '{}'...".format(config.PALABRA_ACTIVACION))

    while True:
        if perro is not None:
            perro.feed()
        act.revisar()
        if _vencido(led_apagar_en):
            led.apagar()
            led_apagar_en = None
        if estado == ESCUCHA and _vencido(escucha_hasta):
            estado = ESPERA
            escucha_hasta = None
            print("Tiempo de escucha agotado. Esperando '{}'...".format(config.PALABRA_ACTIVACION))

        muestras = lector.leer_bloque()
        energia, carga = lector.analizar(muestras)
        segmento = detector.procesar_bloque(energia, carga)

        if segmento is None:
            continue

        if sordo_hasta is not None and not _vencido(sordo_hasta):
            continue

        caracteristicas = feature_extractor.extraer(segmento)
        probs = inferencia.predecir_clases(caracteristicas, m)
        probabilidad = probs[CLASE_ACTIVACION]

        if estado == ESCUCHA:
            # Despues de la frase de activacion: responde preguntas y comandos durante
            # SEGUNDOS_ESCUCHA. Un sonido que no es nada de eso no cierra la escucha.
            pregunta = _mejor_pregunta(probs)
            if pregunta is not None:
                print("[PREGUNTA] {} (confianza {:.2f})".format(pregunta, probs[CLASES.index(pregunta)]))
                _responder(pregunta, lector, perro)
                sordo_hasta = time.ticks_add(time.ticks_ms(), MS_SORDO_TRAS_RESPUESTA)
                estado = ESPERA
                escucha_hasta = None
                print("Esperando '{}'...".format(config.PALABRA_ACTIVACION))
                continue
            comando = _mejor_comando(probs)
            if comando is not None:
                print("[COMANDO] {} (confianza {:.2f})".format(comando, probs[CLASES.index(comando)]))
                comandos.ejecutar(comando, act)
                _hablar(config.RUTA_CLIP_CONFIRMACION, lector, perro)
                sordo_hasta = time.ticks_add(time.ticks_ms(), MS_SORDO_TRAS_RESPUESTA)
                estado = ESPERA
                escucha_hasta = None
                print("Esperando '{}'...".format(config.PALABRA_ACTIVACION))
            else:
                detalle = " ".join("{}={:.2f}".format(CLASES[k], probs[k]) for k in range(len(probs)))
                print("[ESCUCHA] sin pregunta reconocida: {}".format(detalle))
            continue

        # ESPERA: solo la frase de activacion. Las preguntas se ignoran si el asistente no esta activo.
        if probabilidad >= config.UMBRAL_CONFIANZA:
            print("[COMANDO RECONOCIDO] '{}' (confianza {:.2f}).".format(config.PALABRA_ACTIVACION, probabilidad))
            led.encender()
            led_apagar_en = time.ticks_add(time.ticks_ms(), config.SEGUNDOS_LED_ENCENDIDO * 1000)
            _hablar(config.RUTA_CLIP_CONFIRMACION, lector, perro)
            sordo_hasta = time.ticks_add(time.ticks_ms(), MS_SORDO_TRAS_RESPUESTA)
            estado = ESCUCHA
            escucha_hasta = time.ticks_add(time.ticks_ms(), config.SEGUNDOS_ESCUCHA * 1000)
            print("Escuchando {} segundos...".format(config.SEGUNDOS_ESCUCHA))
        else:
            print("[DESCARTADO] sonido no coincide (activacion {:.2f})".format(probabilidad))


if __name__ == "__main__":
    ejecutar()
