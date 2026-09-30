"""Recolecta muestras reales para entrenar el detector de "activar asistente".

Uso (por REPL, con mpremote en Windows Terminal):

    >>> import recolectar_datos as rd
    >>> rd.grabar(1, 20, "cerca")    # 20 veces la frase, a ~30 cm
    >>> rd.grabar(0, 20, "cerca")    # 20 veces OTRA cosa, a ~30 cm
    >>> rd.resumen()                 # cuantas muestras hay por grupo
    >>> rd.deshacer()                # borra la ultima muestra grabada
    >>> rd.liberar()                 # libera el I2S al terminar

etiqueta: 1 = "activar asistente", 0 = cualquier otra cosa (otras frases,
ruidos, tos, aplausos, TV...). distancia: "cerca", "media" o "lejos".

Cada muestra se guarda al instante (una por linea) en config.RUTA_DATASET,
con la secuencia completa de bloques (rms, altas frecuencias, cruces por
cero). Asi, las caracteristicas se pueden recalcular en la PC sin volver
a grabar. Copia el archivo a la PC y usalo con entrenar_modelo.py.
"""

import os

import ujson

import audio_i2s
import config
import feature_extractor
import vad

_estado = {}


def _iniciar():
    if not _estado:
        print("Iniciando I2S (descarta los primeros segundos de arranque)...")
        i2s = audio_i2s.iniciar_i2s()
        _estado["i2s"] = i2s
        _estado["lector"] = audio_i2s.LectorAudio(i2s)
        _estado["detector"] = vad.DetectorVoz()
    return _estado["lector"], _estado["detector"]


def _capturar(lector, detector):
    lector.vaciar_buffer()
    while True:
        muestras = lector.leer_bloque()
        energia, carga = lector.analizar(muestras)
        segmento = detector.procesar_bloque(energia, carga)
        if segmento is not None:
            return segmento


def grabar(etiqueta, n, distancia):
    if etiqueta not in (0, 1):
        raise ValueError("etiqueta debe ser 1 (frase) o 0 (otra cosa)")
    if distancia not in ("cerca", "media", "lejos"):
        raise ValueError("distancia debe ser 'cerca', 'media' o 'lejos'")

    lector, detector = _iniciar()

    if etiqueta == 1:
        print("=== POSITIVAS ({}): di 'activar asistente' con tono natural ===".format(distancia))
    else:
        print("=== NEGATIVAS ({}): di OTRA cosa distinta cada vez ===".format(distancia))
        print("(otras frases de 2-3 palabras, numeros, tu nombre, tos, aplausos, TV...)")

    for i in range(n):
        input("  [Enter] para grabar {}/{}...".format(i + 1, n))
        print("  Escuchando...")
        segmento = _capturar(lector, detector)
        recortado = feature_extractor.recortar(segmento)

        muestra = {
            "bloques": [list(b) for b in segmento],
            "label": etiqueta,
            "distancia": distancia,
        }
        with open(config.RUTA_DATASET, "a") as f:
            f.write(ujson.dumps(muestra) + "\n")

        pico = max(b[0] for b in segmento)
        print("  Guardado: {} bloques ({} utiles), pico {}".format(len(segmento), len(recortado), pico))


def resumen():
    cuentas = {}
    try:
        with open(config.RUTA_DATASET) as f:
            for linea in f:
                m = ujson.loads(linea)
                clave = (m["label"], m["distancia"])
                cuentas[clave] = cuentas.get(clave, 0) + 1
    except OSError:
        print("Todavia no hay muestras en {}.".format(config.RUTA_DATASET))
        return

    total = 0
    for clave in sorted(cuentas):
        etiqueta = "frase" if clave[0] == 1 else "otra "
        print("  {} {:6s} {}".format(etiqueta, clave[1], cuentas[clave]))
        total += cuentas[clave]
    print("  total: {}".format(total))


def deshacer():
    """Borra la ultima muestra (por si la toma salio mal)."""
    ruta_temporal = config.RUTA_DATASET + ".tmp"
    previa = None
    with open(config.RUTA_DATASET) as origen:
        with open(ruta_temporal, "w") as destino:
            for linea in origen:
                if previa is not None:
                    destino.write(previa)
                previa = linea
    os.remove(config.RUTA_DATASET)
    os.rename(ruta_temporal, config.RUTA_DATASET)
    print("Ultima muestra borrada.")


def liberar():
    if _estado:
        _estado["i2s"].deinit()
        _estado.clear()
    print("I2S liberado.")
