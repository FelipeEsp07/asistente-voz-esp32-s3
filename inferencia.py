"""Inferencia en MicroPython puro del modelo exportado por entrenar_modelo.py
(ensamble de CNN 1D multiclase con salida softmax). Sin librerias externas.

predecir_clases() devuelve la probabilidad de cada clase (en el orden de
modelo.CLASES). predecir_probabilidad() devuelve solo la de la frase de
activacion, que es lo que usa asistente.py.
"""

import math

import config
import feature_extractor


def _logits_cnn(x, mb):
    """Conv1D sobre la secuencia (T pasos x C canales) -> maximo y media
    por filtro (+ duracion) -> capa lineal. Mismo calculo que CNN1D en
    entrenar_modelo.py. Devuelve un logit por clase."""
    T = config.PUNTOS_ENVOLVENTE
    C = feature_extractor.NUM_CANALES
    F = len(mb["bc"])
    K = len(mb["Wc"][0]) // C
    L = T - K + 1
    mu, sd = mb["mu"], mb["sd"]

    seq = [[(x[c * T + t] - mu[c]) / sd[c] for c in range(C)] for t in range(T)]
    maximos = [-2.0] * F
    sumas = [0.0] * F
    for t in range(L):
        parche = []
        for k in range(K):
            parche += seq[t + k]
        for f in range(F):
            w = mb["Wc"][f]
            s = mb["bc"][f]
            for i in range(len(parche)):
                s += w[i] * parche[i]
            a = math.tanh(s)
            if a > maximos[f]:
                maximos[f] = a
            sumas[f] += a

    h = maximos + [v / L for v in sumas] + [(x[C * T] - mb["dmu"]) / mb["dsd"]]
    logits = []
    for k in range(len(mb["bd"])):
        z = mb["bd"][k]
        fila = mb["Wd"][k]
        for i in range(len(h)):
            z += fila[i] * h[i]
        logits.append(z)
    return logits


def _softmax(logits):
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def predecir_clases(x, m):
    """x: vector de caracteristicas (feature_extractor.extraer). m: el modulo
    modelo_pesos ya importado (TIPO, CLASES y MIEMBROS). Devuelve la
    probabilidad promedio de cada clase, en el orden de m.CLASES."""
    if m.TIPO != "cnn_multi":
        raise ValueError("Tipo de modelo desconocido: {}".format(m.TIPO))
    total = [0.0] * len(m.CLASES)
    for miembro in m.MIEMBROS:
        for k, p in enumerate(_softmax(_logits_cnn(x, miembro))):
            total[k] += p
    return [t / len(m.MIEMBROS) for t in total]


def predecir_probabilidad(x, m):
    """Probabilidad promedio de que sea la frase de activacion."""
    return predecir_clases(x, m)[m.CLASES.index("activacion")]
