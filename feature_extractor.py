"""Extraccion de caracteristicas de un segmento de voz.

Convierte la secuencia de bloques (rms, hf, zcr) que entrega el VAD en un vector
de tamano fijo para el modelo. Lo usan por igual la ESP32-S3 (main.py) y la PC
(entrenar_modelo.py), asi las caracteristicas de entrenamiento y de uso coinciden.
"""

import math

import config


def _remuestrear(valores, n_puntos):
    """Remuestreo lineal simple de `valores` a `n_puntos`."""
    largo = len(valores)
    if largo == n_puntos:
        return list(valores)
    if largo == 1:
        return [valores[0]] * n_puntos

    salida = []
    for i in range(n_puntos):
        pos = i * (largo - 1) / (n_puntos - 1)
        idx = int(pos)
        frac = pos - idx
        if idx + 1 < largo:
            v = valores[idx] * (1 - frac) + valores[idx + 1] * frac
        else:
            v = valores[idx]
        salida.append(v)
    return salida


def _pico_robusto(amps):
    """Percentil 90 de las amplitudes: un golpe al microfono o un chasquido
    aislado (picos de 10-50x la voz) no debe aplastar la forma de la frase."""
    orden = sorted(amps)
    return orden[int(0.9 * (len(orden) - 1))]


def recortar(bloques):
    """Quita bloques iniciales/finales con amplitud menor a
    RECORTE_FRACCION_PICO * pico. `bloques`: lista de (rms16, hf, zcr).
    Asi la forma no depende de cuanto silencio dejo el VAD al cerrar."""
    amps = [b[0] for b in bloques]
    pico = _pico_robusto(amps)
    if pico <= 0:
        return bloques
    umbral = pico * config.RECORTE_FRACCION_PICO

    inicio = 0
    while inicio < len(amps) - 1 and amps[inicio] < umbral:
        inicio += 1
    fin = len(amps) - 1
    while fin > inicio and amps[fin] < umbral:
        fin -= 1
    return bloques[inicio:fin + 1]


NUM_CANALES = 2  # amplitud (dB) y cruces por cero ponderados


def extraer(bloques):
    """`bloques`: lista de (rms16, hf, zcr), uno por bloque de audio.
    Devuelve 2 * PUNTOS_ENVOLVENTE + 1 valores:
      - amplitud en dB relativa al pico robusto (-40 dB..+4 dB -> 0..1.09):
        la escala logaritmica deja ver las consonantes suaves, que en
        escala lineal quedan tapadas por las vocales,
      - cruces por cero (zcr * 2) ponderados por la amplitud: en los bloques
        casi silenciosos el ruido de fondo no aporta informacion,
      - duracion del segmento recortado relativa a MAX_BLOQUES_PALABRA.
    Se usa igual en la ESP32-S3 y en la PC (entrenar_modelo.py). Elegido por
    validacion cruzada entre ~15 variantes (ver entrenar_modelo.py)."""
    n = config.PUNTOS_ENVOLVENTE
    bloques = recortar(bloques)

    pico = _pico_robusto([b[0] for b in bloques])
    if pico <= 0:
        return [0.0] * (NUM_CANALES * n + 1)
    limite = pico * 1.5
    lineal = [v / pico for v in _remuestrear([min(b[0], limite) for b in bloques], n)]
    zcrs = _remuestrear([min(b[2] * 2, 1.0) for b in bloques], n)

    db = []
    ponderado = []
    for i in range(n):
        a = lineal[i]
        d = 20 * math.log10(a if a > 0.001 else 0.001)
        db.append((max(-40.0, min(4.0, d))) / 40 + 1)
        ponderado.append(zcrs[i] * (a if a < 1.0 else 1.0))

    duracion = len(bloques) / config.MAX_BLOQUES_PALABRA
    return db + ponderado + [duracion]
