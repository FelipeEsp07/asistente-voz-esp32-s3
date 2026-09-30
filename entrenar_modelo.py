"""Evalua y entrena el modelo que detecta "activar asistente".

IMPORTANTE: corre en la PC (numpy), no en la ESP32-S3. Usa el mismo
feature_extractor.py que la placa, asi las caracteristicas son identicas.

Uso:
  1. Graba muestras con recolectar_datos.py y copia dataset.jsonl aqui:
       mpremote connect COM3 cp :dataset.jsonl .
  2. python entrenar_modelo.py            (compara modelos, ~2-4 min)
     python entrenar_modelo.py --rapido   (grilla chica, para probar)
"""

import json
import random
import sys

import numpy as np

import config
import feature_extractor as fe

K_FOLDS = 5
REPETICIONES_CV = 3
VECES_AUMENTO = 6
SEMILLA = 42
T = config.PUNTOS_ENVOLVENTE
DISTANCIAS = ("cerca", "media", "lejos")

NC = fe.NUM_CANALES            # canales: [amplitud dB][zcr ponderado]
D = NC * T + 1                 # + duracion
COL_TODAS = list(range(D))


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


def preparar_datos(datos):  # noqa: D103
    X = np.array([features(d["bloques"]) for d in datos])
    y = np.array([d["label"] for d in datos], dtype=float).reshape(-1, 1)
    dist = np.array([d["distancia"] for d in datos])
    return X, y, dist


def particiones(datos, k, semilla):
    """K particiones estratificadas por (etiqueta, distancia)."""
    rng = random.Random(semilla)
    grupos = {}
    for i, d in enumerate(datos):
        grupos.setdefault((d["label"], d["distancia"]), []).append(i)
    folds = [[] for _ in range(k)]
    for idxs in grupos.values():
        rng.shuffle(idxs)
        for j, i in enumerate(idxs):
            folds[j % k].append(i)
    return folds


def armar_folds(datos, X):
    """Precalcula, para cada repeticion y particion, el entrenamiento
    aumentado y la validacion (sin aumentar). Se comparte entre modelos."""
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
            piezas.append((np.array(Xa), np.array(ya, dtype=float).reshape(-1, 1), val))
        todos.append(piezas)
    return todos


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


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


class MLP:
    tipo = "mlp"

    def __init__(self, ocultas, lam, epocas, lr=0.01, dropout=0.0, semilla=0, columnas=None):
        self.H, self.lam, self.epocas, self.lr = ocultas, lam, epocas, lr
        self.dropout, self.semilla = dropout, semilla
        self.columnas = columnas or COL_TODAS

    def fit(self, X, y):
        X = X[:, self.columnas]
        rng = np.random.default_rng(self.semilla)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-6] = 1.0
        Xn = (X - self.mu) / self.sd
        n, d = Xn.shape
        H = self.H
        p = {"W1": rng.normal(0, (1 / d) ** 0.5, (d, H)), "b1": np.zeros(H),
             "W2": rng.normal(0, (1 / H) ** 0.5, (H, 1)), "b2": np.zeros(1)}
        opt = Adam(p, self.lr)
        for _ in range(self.epocas):
            a1 = np.tanh(Xn @ p["W1"] + p["b1"])
            mask = (rng.random(a1.shape) > self.dropout) / (1 - self.dropout) if self.dropout else 1.0
            ah = a1 * mask
            a2 = sigmoid(ah @ p["W2"] + p["b2"])
            dz2 = (a2 - y) / n
            da1 = (dz2 @ p["W2"].T) * mask
            dz1 = da1 * (1 - a1 ** 2)
            opt.paso({"W2": ah.T @ dz2 + 2 * self.lam / n * p["W2"], "b2": dz2.sum(0),
                      "W1": Xn.T @ dz1 + 2 * self.lam / n * p["W1"], "b1": dz1.sum(0)})
        self.p = p
        return self

    def predict(self, X):
        Xn = (X[:, self.columnas] - self.mu) / self.sd
        return sigmoid(np.tanh(Xn @ self.p["W1"] + self.p["b1"]) @ self.p["W2"] + self.p["b2"])


class CNN1D:
    """Conv1D (K=3) sobre la secuencia (T pasos x NC canales) -> max y media
    global por filtro (+ duracion) -> capa lineal -> sigmoide."""
    tipo = "cnn"

    def __init__(self, filtros, lam, epocas, lr=0.01, K=3, semilla=0):
        self.F, self.lam, self.epocas, self.lr, self.K, self.semilla = filtros, lam, epocas, lr, K, semilla

    @staticmethod
    def _separar(X):
        S = np.stack([X[:, c * T:(c + 1) * T] for c in range(NC)], axis=2)
        return S, X[:, NC * T]

    def _parches(self, S):
        L = S.shape[1] - self.K + 1
        return np.concatenate([S[:, k:k + L, :] for k in range(self.K)], axis=2)

    def _forward(self, S, dur):
        Sn = (S - self.mu) / self.sd
        dn = (dur - self.dmu) / self.dsd
        P = self._parches(Sn)
        A = np.tanh(P @ self.p["Wc"] + self.p["bc"])
        idx = A.argmax(axis=1)
        mx = np.take_along_axis(A, idx[:, None, :], axis=1)[:, 0, :]
        h = np.concatenate([mx, A.mean(axis=1), dn[:, None]], axis=1)
        return P, A, idx, h, sigmoid(h @ self.p["Wd"] + self.p["bd"])

    def fit(self, X, y):
        S, dur = self._separar(X)
        rng = np.random.default_rng(self.semilla)
        self.mu, self.sd = S.mean((0, 1)), S.std((0, 1))
        self.sd[self.sd < 1e-6] = 1.0
        self.dmu, self.dsd = dur.mean(), max(dur.std(), 1e-6)
        n, F, KC = S.shape[0], self.F, self.K * S.shape[2]
        p = {"Wc": rng.normal(0, (1 / KC) ** 0.5, (KC, F)), "bc": np.zeros(F),
             "Wd": rng.normal(0, (1 / (2 * F + 1)) ** 0.5, (2 * F + 1, 1)), "bd": np.zeros(1)}
        self.p = p
        opt = Adam(p, self.lr)
        for _ in range(self.epocas):
            P, A, idx, h, a = self._forward(S, dur)
            L = A.shape[1]
            dz = (a - y) / n
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

def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


def metricas(y, p, dist):
    y, p = y.ravel(), p.ravel()
    pred = p >= 0.5
    m = {"acc": (pred == (y == 1)).mean(), "auc": auc(y, p),
         "tpr": pred[y == 1].mean(), "fpr": pred[y == 0].mean()}
    for d in DISTANCIAS:
        sel = (y == 1) & (dist == d)
        m["tpr_" + d] = pred[sel].mean() if sel.any() else float("nan")
    return m


def evaluar(fabrica, folds, y, dist):
    """fabrica(semilla) -> modelo. Devuelve (metricas promedio, std de acc, probs de la rep 0)."""
    por_rep, primera = [], None
    for r, piezas in enumerate(folds):
        p = np.zeros(len(y))
        for f, (Xa, ya, val) in enumerate(piezas):
            modelo = fabrica(SEMILLA + 10 * r + f).fit(Xa, ya)
            p[val] = modelo.predict(XG[val]).ravel()
        por_rep.append(metricas(y, p, dist))
        if primera is None:
            primera = p.copy()
    prom = {k: float(np.mean([m[k] for m in por_rep])) for k in por_rep[0]}
    return prom, float(np.std([m["acc"] for m in por_rep])), primera


def fila(nombre, m, sd):
    return ("  %-34s acc=%.3f (+-%.3f)  auc=%.3f  tpr=%.2f fpr=%.2f | tpr cerca/media/lejos: %.2f/%.2f/%.2f" % (
        nombre, m["acc"], sd, m["auc"], m["tpr"], m["fpr"], m["tpr_cerca"], m["tpr_media"], m["tpr_lejos"]))


def tabla_umbrales(y, p):
    """(umbral, TPR, FPR) con las probabilidades de validacion cruzada."""
    y, p = y.ravel(), p.ravel()
    return [(u, (p[y == 1] >= u).mean(), (p[y == 0] >= u).mean()) for u in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]


def _lista(a):
    return np.round(np.asarray(a, dtype=float), 6).tolist()


def exportar(ruta, modelo):
    """Escribe modelo_pesos.py (formato que lee inferencia.py en la placa)."""
    miembros = modelo.miembros if isinstance(modelo, Ensamble) else [modelo]
    partes = []
    for m in miembros:
        if m.tipo == "mlp":
            partes.append({"mu": _lista(m.mu), "sd": _lista(m.sd), "W1": _lista(m.p["W1"].T),
                           "B1": _lista(m.p["b1"]), "W2": _lista(m.p["W2"].ravel()), "B2": float(m.p["b2"][0])})
        else:
            partes.append({"mu": _lista(m.mu), "sd": _lista(m.sd), "dmu": float(m.dmu), "dsd": float(m.dsd),
                           "Wc": _lista(m.p["Wc"].T), "bc": _lista(m.p["bc"]),
                           "Wd": _lista(m.p["Wd"].ravel()), "bd": float(m.p["bd"][0])})
    contenido = (
        '"""Modelo entrenado para detectar "activar asistente".\n'
        'Generado por entrenar_modelo.py. No editar a mano.\n'
        '"""\n\n'
        'TIPO = "%s"\nMIEMBROS = [\n' % miembros[0].tipo
        + ",\n".join("    " + repr(m) for m in partes) + "\n]\n")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(contenido)


def main():
    global XG
    rapido = "--rapido" in sys.argv
    datos = cargar(config.RUTA_DATASET)
    XG, y, dist = preparar_datos(datos)
    print("Muestras: %d (%d frase, %d otra). Features por muestra: %d" % (
        len(datos), int(y.sum()), int(len(y) - y.sum()), XG.shape[1]))
    print("Aumento x%d, validacion cruzada %d particiones x %d repeticiones.\n" % (
        VECES_AUMENTO + 1, K_FOLDS, REPETICIONES_CV))

    folds = armar_folds(datos, XG)
    res = {}

    def probar(nombre, fabrica):
        m, sd, p = evaluar(fabrica, folds, y, dist)
        res[nombre] = (m, sd, fabrica, p)
        print(fila(nombre, m, sd), flush=True)

    eps = (300,) if rapido else (150, 300, 600)
    print("--- 1) MLP: grilla de hiperparametros ---")
    for h in ((4, 8) if rapido else (4, 8, 16)):
        for ep in eps:
            probar("mlp h=%d ep=%d" % (h, ep), lambda s, h=h, ep=ep: MLP(h, 0.01, ep, semilla=s))

    print("\n--- 2) CNN 1D: grilla de hiperparametros ---")
    for f in ((4,) if rapido else (4, 8)):
        for ep in eps:
            probar("cnn f=%d ep=%d" % (f, ep), lambda s, f=f, ep=ep: CNN1D(f, 0.01, ep, semilla=s))

    mejor = lambda pref: max((k for k in res if k.startswith(pref)), key=lambda k: (res[k][0]["auc"], res[k][0]["acc"]))
    bm, bc = mejor("mlp "), mejor("cnn ")
    print("\nMejor MLP: %s | Mejor CNN: %s" % (bm, bc))

    print("\n--- 3) Ensambles (5 miembros con bootstrap) de los mejores ---")
    probar("ensamble de " + bm, lambda s, f=res[bm][2]: Ensamble(f, 5, s))
    probar("ensamble de " + bc, lambda s, f=res[bc][2]: Ensamble(f, 5, s))

    ranking = sorted(res, key=lambda k: (-res[k][0]["auc"], -res[k][0]["acc"]))
    print("\n--- Ranking final (por AUC, luego precision) ---")
    for k in ranking[:5]:
        print(fila(k, res[k][0], res[k][1]))

    ganador = ranking[0]
    print("\nGanador: %s" % ganador)
    print("\nUmbral de decision (probabilidades de validacion cruzada del ganador):")
    tabla = tabla_umbrales(y, res[ganador][3])
    for u, tpr, fpr in tabla:
        print("   umbral %.1f -> detecta %3.0f%% de las frases, %3.0f%% de falsos positivos" % (u, 100 * tpr, 100 * fpr))
    umbral = max(tabla, key=lambda t: t[1] - t[2])[0]  # maximiza TPR - FPR (Youden)
    print("Umbral sugerido (maximiza aciertos - falsos positivos): %.1f  -> config.UMBRAL_CONFIANZA" % umbral)

    Xa, ya = [XG[i] for i in range(len(datos))], [d["label"] for d in datos]
    rng = np.random.default_rng(SEMILLA)
    for _ in range(VECES_AUMENTO):
        for d in datos:
            Xa.append(features(aumentar_bloques(d["bloques"], rng)))
            ya.append(d["label"])
    final = res[ganador][2](SEMILLA).fit(np.array(Xa), np.array(ya, dtype=float).reshape(-1, 1))
    exportar("modelo_pesos.py", final)
    print("\nModelo final entrenado con las %d muestras (aumentadas) y exportado a modelo_pesos.py." % len(datos))


if __name__ == "__main__":
    main()
