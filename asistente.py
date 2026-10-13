"""Asistente de voz local para ESP32-S3 + INMP441 + MAX98357A (MicroPython).

Todo corre en el nucleo principal: captura de audio, VAD, deteccion de la
frase de activacion y de comandos, control de actuadores, reproduccion de
respuestas y watchdog. La reproduccion alimenta el watchdog entre bloques,
asi que un clip largo no lo dispara.

Flujo (cascada de 3 decisiones independientes, ver entrenar_modelo.py):
  "Hola Bender" (modelo "activacion") -> confirmacion -> escucha de
  otro/fecha_hora/comandos (modelo "intencion") -> si es fecha_hora, responde
  y vuelve a ESPERA; si es "comandos" (la palabra sola), pasa a escuchar
  SOLO los 4 comandos de actuador (modelo "actuador") -> ejecuta, confirma
  y vuelve a ESPERA.

Cada etapa compite SOLO entre sus propias clases: "intencion" ya no incluye
"activacion" (evita que repetir "Hola Bender" mientras se escucha gane por
probabilidad a una pregunta o un comando real), y "actuador" no comparte
decision con fecha_hora ni con el resto de la frase.

Estados:
- ESPERA: busca la frase de activacion (modelo "activacion").
- ESCUCHA: espera una pregunta o la palabra "comandos" (modelo "intencion"),
  durante config.SEGUNDOS_ESCUCHA.
- ESCUCHA_ACTUADOR: ya se dijo "comandos"; espera uno de los 4 comandos de
  actuador (modelo "actuador"), durante otros config.SEGUNDOS_ESCUCHA.

Requiere modelo_pesos.py generado con entrenar_modelo.py.
"""

import time

from machine import RTC, WDT, Pin

import actuadores
import audio_i2s
import comandos
import config
import espectro
import feature_extractor
import hora
import inferencia
import led
import modelo_pesos as m
import salida_audio
import tono
import vad

ESPERA = "espera"
ESCUCHA = "escucha"
ESCUCHA_ACTUADOR = "escucha_actuador"


def _vencido(instante_ms):
    return instante_ms is not None and time.ticks_diff(time.ticks_ms(), instante_ms) >= 0


CLASES_ACTIVACION = m.MODELOS["activacion"]["clases"]
CLASE_ACTIVACION_SI = CLASES_ACTIVACION.index("activacion")

CLASES_INTENCION = m.MODELOS["intencion"]["clases"]
CLASE_FECHA_HORA = CLASES_INTENCION.index("fecha_hora")
CLASE_COMANDOS = CLASES_INTENCION.index("comandos")

CLASES_ACTUADOR = m.MODELOS["actuador"]["clases"]

ANIO_RELOJ_VALIDO = 2026  # si el reloj esta antes, no se fijo: no se responde la hora
MS_SORDO_TRAS_RESPUESTA = 800  # ms sin escuchar tras cada sonido del asistente (su eco por el parlante)

# Clips de confirmacion que existen hoy, por persona y por genero (ver
# _elegir_clip_confirmacion). Maicol y Cristian todavia no tienen clip
# propio: para ellos se usa el clip de su genero, igual que para cualquier
# hablante que el modelo de "persona" no reconozca con confianza.
CLIP_POR_PERSONA = {"felipe": "confirmacion1.wav", "laura": "confirmacion2.wav"}
CLIP_POR_GENERO = {"hombre": "confirmacion1.wav", "mujer": "confirmacion2.wav"}

# Para la fecha y hora: personas con su propio set de clips de vocabulario en
# clips/ (hoy Felipe y Laura completos). Maicol y Cristian todavia no tienen
# el suyo: para ellos (o para un desconocido) se responde con la voz del
# mismo genero, igual que la confirmacion de arriba.
PERSONA_CON_VOCABULARIO = ("felipe", "laura")
GENERO_A_PERSONA_HORA = {"hombre": "felipe", "mujer": "laura"}


def _hablar(ruta, lector, perro):
    """Reproduce un clip y descarta el eco del microfono al terminar."""
    try:
        salida_audio.reproducir(ruta, alimentar=perro.feed if perro is not None else None)
    except OSError:
        print("[AVISO] No se encontro {}.".format(ruta))
    except ValueError as error:
        print("[AVISO] {} no es un WAV valido: {}".format(ruta, error))
    lector.vaciar_buffer()


def _resolver_actuador(caracteristicas):
    """Comando de actuador mas probable (uno de los 4 de CLASES_ACTUADOR, por
    ejemplo "activar_bombillo") y su confianza, consultando el modelo
    "actuador" sobre el mismo vector de caracteristicas. Solo tiene sentido
    llamarla durante ESCUCHA_ACTUADOR (ya se dijo la palabra "comandos")."""
    probs = inferencia.predecir_grupo(caracteristicas, m, "actuador")
    k = max(range(len(probs)), key=lambda i: probs[i])
    return CLASES_ACTUADOR[k], probs[k]


def _identificar_hablante(caracteristicas, f0, centroide):
    """Persona y genero mas probables, con los modelos "persona" y "genero"
    (cascada ya entrenada, ver modelo_pesos.py). Devuelve (persona,
    confianza_persona, genero).

    `caracteristicas`: el vector de 33 valores de feature_extractor.extraer().
    `f0`: tono de la frase (tono.f0_mediana). `centroide`: centroide
    espectral de la frase (espectro.centroide_frase) -- ambos se calculan
    UNA VEZ por frase (no por bloque, ver espectro.py) y se agregan al final
    para formar el vector de 35 valores que esperan "persona"/"genero"
    (igual que construye preparar_dataset_pc.py en la PC).

    Lanza NotImplementedError si modelo_pesos.py todavia no tiene esos dos
    grupos (entrenamiento en curso): quien llama debe capturarlo."""
    if m.MODELOS.get("persona") is None or m.MODELOS.get("genero") is None:
        raise NotImplementedError("modelo_pesos.py todavia no tiene 'persona'/'genero'")

    x35 = list(caracteristicas) + [f0, centroide]
    probs_persona = inferencia.predecir_grupo(x35, m, "persona")
    probs_genero = inferencia.predecir_grupo(x35, m, "genero")
    clases_persona = m.MODELOS["persona"]["clases"]
    clases_genero = m.MODELOS["genero"]["clases"]
    k_persona = max(range(len(probs_persona)), key=lambda i: probs_persona[i])
    k_genero = max(range(len(probs_genero)), key=lambda i: probs_genero[i])
    return clases_persona[k_persona], probs_persona[k_persona], clases_genero[k_genero]


def _identificar_hablante_seguro(segmento, caracteristicas, lector):
    """Como _identificar_hablante, pero con manejo seguro de errores (tono,
    centroide, inferencia, o modelo_pesos.py sin "persona"/"genero" todavia).
    Devuelve (persona, confianza_persona, genero), o None si no se pudo
    identificar. Identificar a quien habla es una mejora: nunca debe impedir
    que el asistente responda."""
    try:
        muestras_crudas = lector.muestras_de_ultimos_bloques(len(segmento))
        f0 = tono.f0_mediana(muestras_crudas)
        centroide = espectro.centroide_frase(muestras_crudas)
        return _identificar_hablante(caracteristicas, f0, centroide)
    except Exception as error:
        print("[AVISO] No se pudo identificar quien habla: {}".format(error))
        return None


def _elegir_clip_confirmacion(identidad):
    """Clip de confirmacion segun quien hablo (ver CLIP_POR_PERSONA/CLIP_POR_GENERO):
    la voz propia si existe su clip y se reconocio con confianza, si no la del
    genero. `identidad`: lo que devuelve _identificar_hablante_seguro(), o None."""
    if identidad is None:
        return config.RUTA_CLIP_CONFIRMACION
    persona, confianza_persona, genero = identidad
    if confianza_persona >= config.UMBRAL_PERSONA and persona in CLIP_POR_PERSONA:
        ruta = CLIP_POR_PERSONA[persona]
    else:
        ruta = CLIP_POR_GENERO.get(genero, config.RUTA_CLIP_CONFIRMACION)
    print("[QUIEN] persona={} ({:.2f}) genero={} -> {}".format(persona, confianza_persona, genero, ruta))
    return ruta


def _elegir_persona_hora(identidad):
    """Con que voz de clips/ se arma la respuesta de fecha y hora: la propia
    si tiene su set completo (hoy Felipe y Laura) y se reconocio con
    confianza, si no la del genero. `identidad`: lo que devuelve
    _identificar_hablante_seguro(), o None."""
    if identidad is None:
        return hora.PERSONA_POR_DEFECTO
    persona, confianza_persona, genero = identidad
    if confianza_persona >= config.UMBRAL_PERSONA and persona in PERSONA_CON_VOCABULARIO:
        return persona
    print("[QUIEN] persona={} ({:.2f}) genero={} -> voz de {}".format(
        persona, confianza_persona, genero, GENERO_A_PERSONA_HORA.get(genero, hora.PERSONA_POR_DEFECTO)))
    return GENERO_A_PERSONA_HORA.get(genero, hora.PERSONA_POR_DEFECTO)


def _responder(tipo, lector, perro, persona=hora.PERSONA_POR_DEFECTO):
    """Dice la hora, la fecha o ambas uniendo clips de clips/, con la voz de
    `persona` (ver _elegir_persona_hora)."""
    ahora = RTC().datetime()
    if ahora[0] < ANIO_RELOJ_VALIDO:
        print("[AVISO] El reloj no esta fijado ({}). Fijalo desde la PC.".format(ahora[0]))
        return
    rutas = hora.secuencia(tipo, ahora, persona)
    i2s_salida = salida_audio.iniciar_i2s_salida(config.SAMPLE_RATE)
    alimentar = perro.feed if perro is not None else None
    try:
        for ruta in rutas:
            salida_audio.reproducir(ruta, i2s=i2s_salida, alimentar=alimentar)
            time.sleep_ms(config.MS_PAUSA_ENTRE_CLIPS)
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
        if estado != ESPERA and _vencido(escucha_hasta):
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

        if estado == ESCUCHA_ACTUADOR:
            # Ya se dijo la palabra "comandos": aqui SOLO compiten los 4
            # comandos de actuador (modelo "actuador"), sin fecha_hora/otro/activacion.
            etiqueta, confianza = _resolver_actuador(caracteristicas)
            if confianza >= config.UMBRAL_ACTUADOR:
                print("[COMANDO] {} (confianza {:.2f})".format(etiqueta, confianza))
                comandos.ejecutar(etiqueta, act)
                _hablar(config.RUTA_CLIP_CONFIRMACION, lector, perro)
                sordo_hasta = time.ticks_add(time.ticks_ms(), MS_SORDO_TRAS_RESPUESTA)
                estado = ESPERA
                escucha_hasta = None
                print("Esperando '{}'...".format(config.PALABRA_ACTIVACION))
            else:
                print("[ESCUCHA_ACTUADOR] sonido sin reconocer ({} {:.2f}), sigue escuchando".format(
                    etiqueta, confianza))
            continue

        if estado == ESCUCHA:
            # Despues de la frase de activacion: solo compiten otro/fecha_hora/
            # comandos (modelo "intencion"); "activacion" no es una clase aqui
            # (ver entrenar_modelo.py), asi un comando o pregunta real no puede
            # perder contra una repeticion de "Hola Bender".
            probs = inferencia.predecir_grupo(caracteristicas, m, "intencion")
            k = max(range(len(probs)), key=lambda i: probs[i])

            if k == CLASE_FECHA_HORA and probs[k] >= config.UMBRAL_PREGUNTA:
                print("[PREGUNTA] fecha_hora (confianza {:.2f})".format(probs[k]))
                identidad = _identificar_hablante_seguro(segmento, caracteristicas, lector)
                _responder("fecha_hora", lector, perro, _elegir_persona_hora(identidad))
                sordo_hasta = time.ticks_add(time.ticks_ms(), MS_SORDO_TRAS_RESPUESTA)
                estado = ESPERA
                escucha_hasta = None
                print("Esperando '{}'...".format(config.PALABRA_ACTIVACION))
            elif k == CLASE_COMANDOS and probs[k] >= config.UMBRAL_PREGUNTA:
                print("[COMANDOS] se escucho 'comandos' (confianza {:.2f}), escuchando el actuador...".format(probs[k]))
                estado = ESCUCHA_ACTUADOR
                escucha_hasta = time.ticks_add(time.ticks_ms(), config.SEGUNDOS_ESCUCHA * 1000)
            else:
                detalle = " ".join("{}={:.2f}".format(CLASES_INTENCION[i], probs[i]) for i in range(len(probs)))
                print("[ESCUCHA] sonido sin reconocer, sigue escuchando: {}".format(detalle))
            continue

        # ESPERA: solo la frase de activacion (modelo "activacion"). Las preguntas
        # y comandos se ignoran si el asistente no esta activo.
        probs = inferencia.predecir_grupo(caracteristicas, m, "activacion")
        probabilidad = probs[CLASE_ACTIVACION_SI]
        if probabilidad >= config.UMBRAL_CONFIANZA:
            print("[COMANDO RECONOCIDO] '{}' (confianza {:.2f}).".format(config.PALABRA_ACTIVACION, probabilidad))
            led.encender()
            led_apagar_en = time.ticks_add(time.ticks_ms(), config.SEGUNDOS_LED_ENCENDIDO * 1000)
            identidad = _identificar_hablante_seguro(segmento, caracteristicas, lector)
            _hablar(_elegir_clip_confirmacion(identidad), lector, perro)
            sordo_hasta = time.ticks_add(time.ticks_ms(), MS_SORDO_TRAS_RESPUESTA)
            estado = ESCUCHA
            escucha_hasta = time.ticks_add(time.ticks_ms(), config.SEGUNDOS_ESCUCHA * 1000)
            print("Escuchando {} segundos...".format(config.SEGUNDOS_ESCUCHA))
        else:
            print("[DESCARTADO] sonido no coincide (activacion {:.2f})".format(probabilidad))


if __name__ == "__main__":
    ejecutar()
