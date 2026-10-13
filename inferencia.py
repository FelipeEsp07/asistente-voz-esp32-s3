"""Inferencia en MicroPython puro del modelo exportado por entrenar_modelo.py.
Sin librerias externas.

El modelo es una cascada de grupos, cada uno un ensamble de CNN 1D con
salida softmax, evaluados sobre el MISMO vector de caracteristicas:

  - "activacion": "no" / "activacion" (estado ESPERA de asistente.py).
  - "intencion": "otro" / "fecha_hora" / "comandos" (estado ESCUCHA; solo
    tiene sentido si "activacion" dijo que si).
  - "actuador": los 4 comandos de actuador (estado ESCUCHA_ACTUADOR; solo
    tiene sentido si "intencion" dijo "comandos").
  - "persona" / "genero": quien habla (usa ademas el tono, ver asistente.py).

predecir_grupo(x, m, nombre) devuelve la probabilidad de cada clase de ese
grupo, en el orden de m.MODELOS[nombre]["clases"].
"""

import math

import config
import feature_extractor


def _logits_cnn(x, mb):
    """Conv1D sobre la secuencia (T pasos x C canales) -> maximo y media
    por filtro (+ valores extra: duracion, y tono si el grupo lo usa) ->
    capa lineal. Mismo calculo que CNN1D en entrenar_modelo.py. Devuelve un
    logit por clase."""
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

    n_extra = len(mb["emu"])
    extras = [(x[C * T + j] - mb["emu"][j]) / mb["esd"][j] for j in range(n_extra)]
    h = maximos + [v / L for v in sumas] + extras
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


def predecir_grupo(x, m, nombre):
    """x: vector de caracteristicas (feature_extractor.extraer). m: el modulo
    modelo_pesos ya importado (TIPO y MODELOS). nombre: "intencion", "equipo"
    o "verbo". Devuelve la probabilidad promedio de cada clase del grupo, en
    el orden de m.MODELOS[nombre]["clases"]."""
    if m.TIPO != "cnn_cascada":
        raise ValueError("Tipo de modelo desconocido: {}".format(m.TIPO))
    grupo = m.MODELOS[nombre]
    miembros = grupo["miembros"]
    total = [0.0] * len(grupo["clases"])
    for miembro in miembros:
        for k, p in enumerate(_softmax(_logits_cnn(x, miembro))):
            total[k] += p
    return [t / len(miembros) for t in total]
