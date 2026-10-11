"""Evalua y entrena el modelo que clasifica lo que dice la persona.

Clases (en este orden, igual que CLASES de preparar_dataset_pc.py):
  0 otro, 1 activacion ("Hola PAN"), 2 fecha_hora ("fecha y hora").

IMPORTANTE: corre en la PC (numpy), no en la ESP32-S3. Usa el mismo
feature_extractor.py que la placa, asi las caracteristicas son identicas.

Modelo: CNN 1D con salida softmax de varias clases, ensamble de miembros.
Se exporta como modelo_pesos.py (TIPO = "cnn_multi") que lee inferencia.py.

Uso:
  1. Copia dataset.jsonl a esta carpeta: mpremote connect COM3 cp :dataset.jsonl .
  2. python entrenar_modelo.py            (compara configuraciones, unos minutos)
     python entrenar_modelo.py --rapido   (grilla chica, para probar)
"""

import json
import random
import sys
from collections import Counter

import numpy as np

import config
import feature_extractor as fe

K_FOLDS = 5
REPETICIONES_CV = 3
VECES_AUMENTO = 6
SEMILLA = 42
T = config.PUNTOS_ENVOLVENTE

CLASES = ("otro", "activacion", "fecha_hora", "activar_motobomba", "activar_bombillo", "apagar_bombillo",
          "apagar_motobomba", "activar_secador", "apagar_secador")
K = len(CLASES)
IDX_ACTIVACION = CLASES.index("activacion")

NC = fe.NUM_CANALES            # canales: [amplitud dB][zcr ponderado]
D = NC * T + 1                 # + duracion


# --------------------------------------------------------------------------
# Datos y aumento (sobre la secuencia de bloques, antes de extraer features)
# --------------------------------------------------------------------------

def cargar(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def aumentar_bloques(bloques, rng):
    """Variante realista de un segmento: bordes distintos (el VAD no corta
    siempre igual), estiramiento temporal (hablar mas rapido/lento) y ruido."""
    ini, fin = rng.integers(0, 3), rng.integers(0, 3)
    if len(bloques) - ini - fin >= 6:
        bloques = bloques[ini:len(bloques) - fin]
    arr = np.array(bloques, dtype=float)
    m = max(6, int(round(len(arr) * rng.uniform(0.85, 1.15))))
    idx = np.linspace(0, len(arr) - 1, m)
    est = np.stack([np.interp(idx, np.arange(len(arr)), arr[:, c]) for c in range(3)], axis=1)  # rms, hf, zcr
    est[:, 0] *= rng.normal(1, 0.08, size=m)
    est[:, 1] = np.clip(est[:, 1] + rng.normal(0, 0.03, size=m), 0, None)
    est[:, 2] = np.clip(est[:, 2] + rng.normal(0, 0.01, size=m), 0, 1)
    return [tuple(r) for r in est]


def features(bloques):
    return fe.extraer([tuple(b) for b in bloques])


def preparar_datos(datos):
    X = np.array([features(d["bloques"]) for d in datos])
    y = np.array([d["label"] for d in datos], dtype=int)
    return X, y


def particiones(datos, k, semilla):
    """K particiones estratificadas por etiqueta."""
    rng = random.Random(semilla)
    grupos = {}
    for i, d in enumerate(datos):
        grupos.setdefault(d["label"], []).append(i)
    folds = [[] for _ in range(k)]
    for idxs in grupos.values():
        rng.shuffle(idxs)
        for j, i in enumerate(idxs):
            folds[j % k].append(i)
    return folds


def armar_folds(datos, X):
    """Precalcula, para cada repeticion y particion, el entrenamiento
    aumentado y la validacion (sin aumentar). Se comparte entre configuraciones."""
    todos = []
    for rep in range(REPETICIONES_CV):
        folds = particiones(datos, K_FOLDS, SEMILLA + rep)
        rng = np.random.default_rng(SEMILLA + 100 * rep)
        piezas = []
        for f in range(K_FOLDS):
            val = folds[f]
            tr = [i for g in range(K_FOLDS) if g != f for i in folds[g]]
            Xa, ya = [X[i] for i in tr], [datos[i]["label"] for i in tr]
            for _ in range(VECES_AUMENTO):
                for i in tr:
                    Xa.append(features(aumentar_bloques(datos[i]["bloques"], rng)))
                    ya.append(datos[i]["label"])
            piezas.append((np.array(Xa), np.array(ya, dtype=int), val))
        todos.append(piezas)
    return todos


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Adam:
    def __init__(self, params, lr):
        self.p, self.lr, self.t = params, lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def paso(self, grads):
        self.t += 1
        for k in self.p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * grads[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


class CNN1D:
    """Conv1D (K=3) sobre la secuencia (T pasos x NC canales) -> max y media
    global por filtro (+ duracion) -> capa lineal -> softmax de K clases."""

    def __init__(self, filtros, lam, epocas, lr=0.01, K_conv=3, semilla=0):
        self.F, self.lam, self.epocas, self.lr = filtros, lam, epocas, lr
        self.K_conv, self.semilla = K_conv, semilla

    @staticmethod
    def _separar(X):
        S = np.stack([X[:, c * T:(c + 1) * T] for c in range(NC)], axis=2)
        return S, X[:, NC * T]

    def _parches(self, S):
        L = S.shape[1] - self.K_conv + 1
        return np.concatenate([S[:, k:k + L, :] for k in range(self.K_conv)], axis=2)

    def _forward(self, S, dur):
        Sn = (S - self.mu) / self.sd
        dn = (dur - self.dmu) / self.dsd
        P = self._parches(Sn)
        A = np.tanh(P @ self.p["Wc"] + self.p["bc"])
        idx = A.argmax(axis=1)
        mx = np.take_along_axis(A, idx[:, None, :], axis=1)[:, 0, :]
        h = np.concatenate([mx, A.mean(axis=1), dn[:, None]], axis=1)
        return P, A, idx, h, softmax(h @ self.p["Wd"] + self.p["bd"])

    def fit(self, X, y):
        S, dur = self._separar(X)
        rng = np.random.default_rng(self.semilla)
        self.mu, self.sd = S.mean((0, 1)), S.std((0, 1))
        self.sd[self.sd < 1e-6] = 1.0
        self.dmu, self.dsd = dur.mean(), max(dur.std(), 1e-6)
        Y = np.eye(K)[y]
        n, F, KC = S.shape[0], self.F, self.K_conv * S.shape[2]
        p = {"Wc": rng.normal(0, (1 / KC) ** 0.5, (KC, F)), "bc": np.zeros(F),
             "Wd": rng.normal(0, (1 / (2 * F + 1)) ** 0.5, (2 * F + 1, K)), "bd": np.zeros(K)}
        self.p = p
        opt = Adam(p, self.lr)
        for _ in range(self.epocas):
            P, A, idx, h, pr = self._forward(S, dur)
            L = A.shape[1]
            dz = (pr - Y) / n                       # gradiente de entropia cruzada con softmax
            dh = dz @ p["Wd"].T
            dA = np.zeros_like(A)
            np.put_along_axis(dA, idx[:, None, :], dh[:, None, :F], axis=1)
            dA += dh[:, None, F:2 * F] / L
            dZ = dA * (1 - A ** 2)
            opt.paso({"Wd": h.T @ dz + 2 * self.lam / n * p["Wd"], "bd": dz.sum(0),
                      "Wc": P.reshape(-1, KC).T @ dZ.reshape(-1, F) + 2 * self.lam / n * p["Wc"],
                      "bc": dZ.sum((0, 1))})
        return self

    def predict(self, X):
        """Probabilidades (n, K), una columna por clase de CLASES."""
        S, dur = self._separar(X)
        return self._forward(S, dur)[4]


class Ensamble:
    """Promedia M modelos entrenados con remuestreo bootstrap y semillas distintas."""

    def __init__(self, fabrica, M=5, semilla=0):
        self.fabrica, self.M, self.semilla = fabrica, M, semilla

    def fit(self, X, y):
        rng = np.random.default_rng(self.semilla)
        self.miembros = []
        for i in range(self.M):
            idx = rng.integers(0, len(X), len(X))
            self.miembros.append(self.fabrica(self.semilla + i + 1).fit(X[idx], y[idx]))
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.miembros], axis=0)


# --------------------------------------------------------------------------
# Evaluacion
# --------------------------------------------------------------------------

def auc(es_positivo, p):
    pos, neg = p[es_positivo], p[~es_positivo]
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


def metricas(y, P):
    """Metricas multiclase. La frase de activacion tambien se evalua como
    problema binario (activacion contra el resto), que es lo que usa la placa."""
    pred = P.argmax(axis=1)
    m = {"acc": (pred == y).mean()}
    for k, nombre in enumerate(CLASES):
        sel = y == k
        m["rec_" + nombre] = (pred[sel] == k).mean() if sel.any() else float("nan")
    es_act = y == IDX_ACTIVACION
    m["auc_act"] = auc(es_act, P[:, IDX_ACTIVACION])
    m["tpr_act"] = (pred[es_act] == IDX_ACTIVACION).mean()
    m["fpr_act"] = (pred[~es_act] == IDX_ACTIVACION).mean()
    return m


def evaluar(fabrica, folds, X, y):
    """fabrica(semilla) -> modelo. Devuelve (metricas promedio, std de acc, probs de la rep 0)."""
    por_rep, primera = [], None
    for r, piezas in enumerate(folds):
        P = np.zeros((len(y), K))
        for f, (Xa, ya, val) in enumerate(piezas):
            modelo = fabrica(SEMILLA + 10 * r + f).fit(Xa, ya)
            P[val] = modelo.predict(X[val])
        por_rep.append(metricas(y, P))
        if primera is None:
            primera = P.copy()
    prom = {k: float(np.nanmean([m[k] for m in por_rep])) for k in por_rep[0]}
    return prom, float(np.std([m["acc"] for m in por_rep])), primera


def fila(nombre, m, sd):
    recalls = " ".join("%s=%.2f" % (c, m["rec_" + c]) for c in CLASES)
    return "  %-30s acc=%.3f (+-%.3f) | %s | activacion: auc=%.3f tpr=%.2f fpr=%.2f" % (
        nombre, m["acc"], sd, recalls, m["auc_act"], m["tpr_act"], m["fpr_act"])


def tabla_umbrales(y, P):
    """(umbral, TPR, FPR) de la clase activacion con probabilidades de validacion cruzada."""
    es_act, pa = y == IDX_ACTIVACION, P[:, IDX_ACTIVACION]
    return [(u, (pa[es_act] >= u).mean(), (pa[~es_act] >= u).mean()) for u in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]


def _lista(a):
    return np.round(np.asarray(a, dtype=float), 6).tolist()


def exportar(ruta, modelo):
    """Escribe modelo_pesos.py (formato que lee inferencia.py en la placa)."""
    miembros = modelo.miembros if isinstance(modelo, Ensamble) else [modelo]
    partes = [{"mu": _lista(m.mu), "sd": _lista(m.sd), "dmu": float(m.dmu), "dsd": float(m.dsd),
               "Wc": _lista(m.p["Wc"].T), "bc": _lista(m.p["bc"]),
               "Wd": _lista(m.p["Wd"].T), "bd": _lista(m.p["bd"])} for m in miembros]
    contenido = (
        '"""Modelo entrenado para clasificar lo que dice la persona.\n'
        'Generado por entrenar_modelo.py. No editar a mano.\n'
        '"""\n\n'
        'TIPO = "cnn_multi"\n'
        'CLASES = %r\n' % (CLASES,)
        + 'MIEMBROS = [\n' + ",\n".join("    " + repr(m) for m in partes) + "\n]\n")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(contenido)


def main():
    rapido = "--rapido" in sys.argv
    datos = cargar(config.RUTA_DATASET)
    X, y = preparar_datos(datos)
    conteo = Counter(y.tolist())
    print("Muestras: %d. Features por muestra: %d" % (len(datos), X.shape[1]))
    for k, nombre in enumerate(CLASES):
        print("  %-14s %3d" % (nombre, conteo.get(k, 0)))
    print("Aumento x%d, validacion cruzada %d particiones x %d repeticiones.\n" % (
        VECES_AUMENTO + 1, K_FOLDS, REPETICIONES_CV))

    folds = armar_folds(datos, X)
    res = {}

    def probar(nombre, fabrica):
        m, sd, p = evaluar(fabrica, folds, X, y)
        res[nombre] = (m, sd, fabrica, p)
        print(fila(nombre, m, sd), flush=True)

    eps = (300,) if rapido else (150, 300, 600)
    print("--- 1) CNN 1D multiclase: grilla de hiperparametros ---")
    for f in ((4,) if rapido else (4, 8)):
        for ep in eps:
            probar("cnn f=%d ep=%d" % (f, ep), lambda s, f=f, ep=ep: CNN1D(f, 0.01, ep, semilla=s))

    mejor = max(res, key=lambda k: (res[k][0]["acc"], res[k][0]["auc_act"]))
    print("\nMejor configuracion individual: %s" % mejor)

    print("\n--- 2) Ensamble (5 miembros con bootstrap) de la mejor ---")
    probar("ensamble de " + mejor, lambda s, f=res[mejor][2]: Ensamble(f, 5, s))

    ranking = sorted(res, key=lambda k: (-res[k][0]["acc"], -res[k][0]["auc_act"]))
    print("\n--- Ranking final (por precision multiclase, luego AUC de activacion) ---")
    for k in ranking[:5]:
        print(fila(k, res[k][0], res[k][1]))

    ganador = ranking[0]
    print("\nGanador: %s" % ganador)
    print("\nUmbral de activacion (probabilidades de validacion cruzada del ganador):")
    tabla = tabla_umbrales(y, res[ganador][3])
    for u, tpr, fpr in tabla:
        print("   umbral %.1f -> detecta %3.0f%% de las frases, %3.0f%% de falsos positivos" % (u, 100 * tpr, 100 * fpr))
    umbral = max(tabla, key=lambda t: t[1] - t[2])[0]  # maximiza TPR - FPR (Youden)
    print("Umbral sugerido: %.1f  -> config.UMBRAL_CONFIANZA" % umbral)

    Xa, ya = [X[i] for i in range(len(datos))], [d["label"] for d in datos]
    rng = np.random.default_rng(SEMILLA)
    for _ in range(VECES_AUMENTO):
        for d in datos:
            Xa.append(features(aumentar_bloques(d["bloques"], rng)))
            ya.append(d["label"])
    final = res[ganador][2](SEMILLA).fit(np.array(Xa), np.array(ya, dtype=int))
    exportar("modelo_pesos.py", final)
    print("\nModelo final entrenado con las %d muestras (aumentadas) y exportado a modelo_pesos.py." % len(datos))


if __name__ == "__main__":
    main()
