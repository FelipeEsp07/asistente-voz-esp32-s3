"""Inferencia en MicroPython puro del modelo exportado por entrenar_modelo.py
(ensamble de CNN 1D o de MLP, promediando la probabilidad de cada miembro).
Sin librerias externas.
"""

import math

import config
import feature_extractor


def _sigmoid(z):
    return 1 / (1 + math.exp(-z))


def _predecir_mlp(x, mb):
    entradas = [(x[i] - mb["mu"][i]) / mb["sd"][i] for i in range(len(x))]
    z = mb["B2"]
    for j in range(len(mb["B1"])):
        s = mb["B1"][j]
        fila = mb["W1"][j]
        for i in range(len(entradas)):
            s += fila[i] * entradas[i]
        z += mb["W2"][j] * math.tanh(s)
    return _sigmoid(z)


def _predecir_cnn(x, mb):
    """Conv1D sobre la secuencia (T pasos x C canales) -> maximo y media por
    filtro (+ duracion) -> capa lineal. Mismo calculo que CNN1D en
    entrenar_modelo.py."""
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
    z = mb["bd"]
    for i in range(len(h)):
        z += mb["Wd"][i] * h[i]
    return _sigmoid(z)


def predecir_probabilidad(x, m):
    """x: vector de caracteristicas (feature_extractor.extraer). m: el modulo
    modelo_pesos ya importado (TIPO y MIEMBROS). Devuelve la probabilidad
    promedio de que sea la frase de activacion."""
    if m.TIPO == "cnn":
        predecir = _predecir_cnn
    elif m.TIPO == "mlp":
        predecir = _predecir_mlp
    else:
        raise ValueError("Tipo de modelo desconocido: {}".format(m.TIPO))

    total = 0.0
    for miembro in m.MIEMBROS:
        total += predecir(x, miembro)
    return total / len(m.MIEMBROS)
