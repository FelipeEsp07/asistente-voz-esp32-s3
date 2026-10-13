"""Evalua y entrena el modelo que clasifica lo que dice la persona.

Arquitectura en cascada (misma grabacion, varias decisiones mas faciles en
vez de una sola decision de muchas clases parecidas entre si). Cada etapa
solo corre si la anterior la activo, y cada una compite SOLO entre las
clases que tiene sentido confundir en ese momento:

  1. ACTIVACION (todas las muestras): es o no es "Hola Bender". Se evalua en
     el estado ESPERA de asistente.py. El negativo incluye TODO lo demas
     (otro, fecha_hora, comandos, frases de actuador), asi aprende a
     distinguir la frase de activacion de cualquier cosa, no solo del ruido.
  2. INTENCION (una vez detectada la activacion, estado ESCUCHA): "otro",
     "fecha_hora" o "comandos" (la palabra "comandos" sola, NO la frase
     del actuador). "activacion" ya no es una clase aqui -- si se repite
     "Hola Bender" mientras se escucha, se etiqueta como "otro" al entrenar,
     para que no compita por probabilidad con una pregunta o un comando real
     (ese era el problema: antes "activacion" ganaba la discusion y una
     pregunta de hora o un comando se perdian).
  3. ACTUADOR (solo si INTENCION dijo "comandos", estado ESCUCHA_ACTUADOR):
     cual de los 4 comandos se dijo -- "activar_bombillo", "apagar_bombillo",
     "activar_secador" o "apagar_secador". Antes esto eran dos clasificadores
     binarios encadenados (EQUIPO y VERBO); ahora es uno solo de 4 clases
     entrenado directamente, para no multiplicar el error de dos decisiones
     encadenadas.

En la placa se corre cada etapa sobre el MISMO vector de caracteristicas.
Esto reparte mejor las pocas muestras que hay por clase y evita que una
etapa "vea" clases que no le corresponden.

IMPORTANTE: corre en la PC (numpy), no en la ESP32-S3. Usa el mismo
feature_extractor.py que la placa, asi las caracteristicas son identicas.

Se exporta como modelo_pesos.py (TIPO = "cnn_cascada") que lee inferencia.py.

Uso:
  python entrenar_modelo.py            (compara configuraciones, unos minutos)
  python entrenar_modelo.py --rapido   (grilla chica, para probar)
"""

import json
import random
import sys

import numpy as np

import config
import feature_extractor as fe

K_FOLDS = 5
REPETICIONES_CV = 5  # con pocas muestras por clase, 3 repeticiones variaban mucho entre corridas
VECES_AUMENTO = 6
SEMILLA = 42
T = config.PUNTOS_ENVOLVENTE

# Clases "finas" tal como las etiqueta preparar_dataset_pc.py. "comandos" se
# agrego AL FINAL (indice 11) para no tener que reetiquetar las muestras de
# actuador ya guardadas en dataset.jsonl (indices 3 a 10, sin cambios).
FINE_CLASES = ("otro", "activacion", "fecha_hora", "activar_motobomba", "activar_bombillo", "apagar_bombillo",
               "apagar_motobomba", "activar_secador", "apagar_secador", "activar_maquinilla", "apagar_maquinilla",
               "comandos")
# Fina 3 a 10 (exclusivo el fin): frases completas de actuador ("activar
# bombillo", etc.), una por combinacion de equipo+verbo. Ya NO entran a
# "intencion": solo se usan para entrenar "actuador" (paso 3 de la cascada).
IDX_FINA_ACTUADOR_INI = 3
IDX_FINA_ACTUADOR_FIN = 11
IDX_FINA_PALABRA_COMANDOS = FINE_CLASES.index("comandos")  # 11: solo la palabra "comandos"

ACTIVACION_CLASES = ("no", "activacion")
INTENCION_CLASES = ("otro", "fecha_hora", "comandos")
# Por ahora solo bombillo y secador (motobomba y maquinilla quedan pausados).
ACTUADOR_CLASES = ("activar_bombillo", "apagar_bombillo", "activar_secador", "apagar_secador")

# Quien habla (de las 4 personas grabadas) y su genero, para la voz de respuesta
# cuando quien habla no es ninguna de las 4 (ver asistente.py).
PERSONA_CLASES = ("felipe", "laura", "maicol", "cristian")
GENERO_CLASES = ("hombre", "mujer")
GENERO_DE_PERSONA = {"felipe": "hombre", "laura": "mujer", "maicol": "hombre", "cristian": "hombre"}

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
    y_fina = np.array([d["label"] for d in datos], dtype=int)
    return X, y_fina


def etiquetas_activacion(y_fina):
    """1 si la frase es "Hola Bender" (fina 'activacion'), 0 en cualquier otro
    caso (incluye otro, fecha_hora, comandos y las frases de actuador)."""
    idx_act = FINE_CLASES.index("activacion")
    return (y_fina == idx_act).astype(int)


def etiquetas_intencion(y_fina):
    """Solo se llama con muestras que NO son de actuador (ver
    IDX_FINA_ACTUADOR_INI/FIN en quien llama). "activacion" se etiqueta como
    "otro": repetir "Hola Bender" durante la escucha debe ignorarse igual que
    cualquier sonido que no sea una pregunta ni la palabra "comandos", no
    competir por probabilidad contra ellas."""
    idx_otro = FINE_CLASES.index("otro")
    idx_act = FINE_CLASES.index("activacion")
    idx_fecha = FINE_CLASES.index("fecha_hora")
    mapa = {idx_otro: INTENCION_CLASES.index("otro"), idx_act: INTENCION_CLASES.index("otro"),
            idx_fecha: INTENCION_CLASES.index("fecha_hora"),
            IDX_FINA_PALABRA_COMANDOS: INTENCION_CLASES.index("comandos")}
    return np.array([mapa[int(f)] for f in y_fina])


def etiquetas_actuador(y_fina_actuador):
    """Para muestras de actuador: la etiqueta fina YA es el nombre de la
    clase de actuador ("activar_bombillo", etc.), es una traduccion directa
    de nombre a indice."""
    return np.array([ACTUADOR_CLASES.index(FINE_CLASES[i]) for i in y_fina_actuador])


def balancear_submuestreo(y, limite_por_clase, semilla):
    """Indices que dejan cada clase con a lo sumo `limite_por_clase` muestras,
    descartando el resto al azar. Sirve para que una clase con muchas mas
    muestras que las demas no domine el entrenamiento. No modifica el
    dataset en disco, solo el subconjunto usado para esta ronda de
    entrenamiento."""
    rng = random.Random(semilla)
    grupos = {}
    for i, etiqueta in enumerate(y):
        grupos.setdefault(int(etiqueta), []).append(i)
    seleccion = []
    for idxs in grupos.values():
        rng.shuffle(idxs)
        seleccion.extend(idxs[:limite_por_clase])
    seleccion.sort()
    return seleccion


def particiones(y, k, semilla):
    """K particiones estratificadas por etiqueta."""
    rng = random.Random(semilla)
    grupos = {}
    for i, etiqueta in enumerate(y):
        grupos.setdefault(int(etiqueta), []).append(i)
    folds = [[] for _ in range(k)]
    for idxs in grupos.values():
        rng.shuffle(idxs)
        for j, i in enumerate(idxs):
            folds[j % k].append(i)
    return folds


def armar_folds(datos, X, y, extra=None):
    """Precalcula, para cada repeticion y particion, el entrenamiento
    aumentado y la validacion (sin aumentar). Se comparte entre configuraciones.
    `datos` y `X` deben tener el mismo orden que `y` (pueden ser un subconjunto,
    como las muestras de actuador para entrenar el grupo "actuador").

    `extra`, si se da, es un arreglo (n, k) con valores que no se recalculan al
    aumentar (el tono no cambia solo por estirar el audio un poco): se copian
    tal cual en cada version aumentada de la muestra."""
    todos = []
    for rep in range(REPETICIONES_CV):
        folds = particiones(y, K_FOLDS, SEMILLA + rep)
        rng = np.random.default_rng(SEMILLA + 100 * rep)
        piezas = []
        for f in range(K_FOLDS):
            val = folds[f]
            tr = [i for g in range(K_FOLDS) if g != f for i in folds[g]]
            Xa, ya = [X[i] for i in tr], [int(y[i]) for i in tr]
            for _ in range(VECES_AUMENTO):
                for i in tr:
                    vec = features(aumentar_bloques(datos[i]["bloques"], rng))
                    if extra is not None:
                        vec = np.concatenate([vec, extra[i]])
                    Xa.append(vec)
                    ya.append(int(y[i]))
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
    global por filtro (+ valores extra, p. ej. duracion y tono) -> capa lineal
    -> softmax de n_clases. El numero de valores extra se toma de X: todo lo
    que sobra despues de los NC*T puntos de la envolvente."""

    def __init__(self, filtros, lam, epocas, n_clases, lr=0.01, K_conv=3, semilla=0):
        self.F, self.lam, self.epocas, self.lr = filtros, lam, epocas, lr
        self.n_clases, self.K_conv, self.semilla = n_clases, K_conv, semilla

    @staticmethod
    def _separar(X):
        S = np.stack([X[:, c * T:(c + 1) * T] for c in range(NC)], axis=2)
        extras = X[:, NC * T:]  # duracion, tono, ... (lo que agregue quien llama)
        return S, extras

    def _parches(self, S):
        L = S.shape[1] - self.K_conv + 1
        return np.concatenate([S[:, k:k + L, :] for k in range(self.K_conv)], axis=2)

    def _forward(self, S, extras):
        Sn = (S - self.mu) / self.sd
        en = (extras - self.emu) / self.esd
        P = self._parches(Sn)
        A = np.tanh(P @ self.p["Wc"] + self.p["bc"])
        idx = A.argmax(axis=1)
        mx = np.take_along_axis(A, idx[:, None, :], axis=1)[:, 0, :]
        h = np.concatenate([mx, A.mean(axis=1), en], axis=1)
        return P, A, idx, h, softmax(h @ self.p["Wd"] + self.p["bd"])

    def fit(self, X, y):
        S, extras = self._separar(X)
        rng = np.random.default_rng(self.semilla)
        self.mu, self.sd = S.mean((0, 1)), S.std((0, 1))
        self.sd[self.sd < 1e-6] = 1.0
        self.emu, self.esd = extras.mean(0), extras.std(0)
        self.esd[self.esd < 1e-6] = 1.0
        n_extra = extras.shape[1]
        Y = np.eye(self.n_clases)[y]
        n, F, KC = S.shape[0], self.F, self.K_conv * S.shape[2]
        p = {"Wc": rng.normal(0, (1 / KC) ** 0.5, (KC, F)), "bc": np.zeros(F),
             "Wd": rng.normal(0, (1 / (2 * F + n_extra)) ** 0.5, (2 * F + n_extra, self.n_clases)),
             "bd": np.zeros(self.n_clases)}
        self.p = p
        opt = Adam(p, self.lr)
        for _ in range(self.epocas):
            P, A, idx, h, pr = self._forward(S, extras)
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
        """Probabilidades (n, n_clases)."""
        S, extras = self._separar(X)
        return self._forward(S, extras)[4]


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


def metricas(y, P, clases):
    """Precision multiclase y recall por clase."""
    pred = P.argmax(axis=1)
    m = {"acc": (pred == y).mean()}
    for k, nombre in enumerate(clases):
        sel = y == k
        m["rec_" + nombre] = (pred[sel] == k).mean() if sel.any() else float("nan")
    return m


def evaluar(fabrica, folds, X, y, clases):
    """fabrica(semilla) -> modelo. Devuelve (metricas promedio, std de acc, probs de la rep 0)."""
    por_rep, primera = [], None
    for r, piezas in enumerate(folds):
        P = np.zeros((len(y), len(clases)))
        for f, (Xa, ya, val) in enumerate(piezas):
            modelo = fabrica(SEMILLA + 10 * r + f).fit(Xa, ya)
            P[val] = modelo.predict(X[val])
        por_rep.append(metricas(y, P, clases))
        if primera is None:
            primera = P.copy()
    prom = {k: float(np.nanmean([m[k] for m in por_rep])) for k in por_rep[0]}
    return prom, float(np.std([m["acc"] for m in por_rep])), primera


def fila(nombre, m, sd, clases):
    recalls = " ".join("%s=%.2f" % (c, m["rec_" + c]) for c in clases)
    return "  %-30s acc=%.3f (+-%.3f) | %s" % (nombre, m["acc"], sd, recalls)


def tabla_umbrales_clase(y, P, idx_clase, umbrales=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """(umbral, TPR, FPR) de una clase especifica, binarizando el resto."""
    es, pa = y == idx_clase, P[:, idx_clase]
    return [(u, (pa[es] >= u).mean(), (pa[~es] >= u).mean()) for u in umbrales]


def _lista(a):
    return np.round(np.asarray(a, dtype=float), 6).tolist()


def entrenar_grupo(nombre, datos, X, y, clases, rapido, extra=None):
    """Grilla de CNN1D + ensamble para un grupo (intencion, equipo, verbo,
    persona o genero). `extra`: ver armar_folds. Devuelve (modelo final
    entrenado con todos los datos, metricas CV del ganador, probabilidades CV
    del ganador)."""
    print("\n=== %s: %d muestras, %d clases ===" % (nombre, len(y), len(clases)))
    for k, c in enumerate(clases):
        print("  %-16s %3d" % (c, int((y == k).sum())))

    folds = armar_folds(datos, X, y, extra)
    res = {}

    def probar(etiqueta, fabrica):
        m, sd, p = evaluar(fabrica, folds, X, y, clases)
        res[etiqueta] = (m, sd, fabrica, p)
        print(fila(etiqueta, m, sd, clases), flush=True)

    eps = (300,) if rapido else (150, 300, 600)
    for f in ((4,) if rapido else (4, 8)):
        for ep in eps:
            probar("cnn f=%d ep=%d" % (f, ep),
                   lambda s, f=f, ep=ep: CNN1D(f, 0.01, ep, len(clases), semilla=s))

    mejor = max(res, key=lambda k: res[k][0]["acc"])
    probar("ensamble de " + mejor, lambda s, fab=res[mejor][2]: Ensamble(fab, 5, s))

    ranking = sorted(res, key=lambda k: -res[k][0]["acc"])
    ganador = ranking[0]
    print("Ganador de '%s': %s (acc=%.3f)" % (nombre, ganador, res[ganador][0]["acc"]))

    Xa, ya = [X[i] for i in range(len(datos))], [int(y[i]) for i in range(len(datos))]
    rng = np.random.default_rng(SEMILLA)
    for _ in range(VECES_AUMENTO):
        for i, d in enumerate(datos):
            vec = features(aumentar_bloques(d["bloques"], rng))
            if extra is not None:
                vec = np.concatenate([vec, extra[i]])
            Xa.append(vec)
            ya.append(int(y[i]))
    final = res[ganador][2](SEMILLA).fit(np.array(Xa), np.array(ya, dtype=int))
    return final, res[ganador][0], res[ganador][3]


def exportar_cascada(ruta, grupos):
    """Escribe modelo_pesos.py (formato que lee inferencia.py en la placa).
    grupos: {nombre: (modelo, clases)}."""
    bloques = []
    for nombre, (modelo, clases) in grupos.items():
        miembros = modelo.miembros if isinstance(modelo, Ensamble) else [modelo]
        partes = [{"mu": _lista(m.mu), "sd": _lista(m.sd), "emu": _lista(m.emu), "esd": _lista(m.esd),
                   "Wc": _lista(m.p["Wc"].T), "bc": _lista(m.p["bc"]),
                   "Wd": _lista(m.p["Wd"].T), "bd": _lista(m.p["bd"])} for m in miembros]
        cuerpo = ",\n".join("            " + repr(p) for p in partes)
        bloques.append(
            "    %r: {\n        \"clases\": %r,\n        \"miembros\": [\n%s\n        ],\n    },\n"
            % (nombre, clases, cuerpo)
        )
    contenido = (
        '"""Modelo entrenado para clasificar lo que dice la persona, en cascada.\n'
        'Generado por entrenar_modelo.py. No editar a mano.\n\n'
        'MODELOS["activacion"] decide si es "Hola Bender" (estado ESPERA). Si lo es,\n'
        'MODELOS["intencion"] decide "otro"/"fecha_hora"/"comandos" (estado ESCUCHA).\n'
        'Si es "comandos", MODELOS["actuador"] decide cual de los 4 comandos de\n'
        'actuador se dijo (estado ESCUCHA_ACTUADOR). Los tres grupos se evaluan sobre\n'
        'el mismo vector de caracteristicas. Ver asistente.py para el flujo completo.\n'
        '"""\n\n'
        'TIPO = "cnn_cascada"\n'
        'MODELOS = {\n' + "".join(bloques) + '}\n'
    )
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(contenido)


def main():
    rapido = "--rapido" in sys.argv
    datos = cargar(config.RUTA_DATASET)
    X, y_fina = preparar_datos(datos)
    print("Muestras totales: %d. Features por muestra: %d" % (len(datos), X.shape[1]))
    print("Aumento x%d, validacion cruzada %d particiones x %d repeticiones." % (
        VECES_AUMENTO + 1, K_FOLDS, REPETICIONES_CV))

    # --- 1) Activacion: binaria, con TODAS las muestras balanceadas 50/50
    # (el negativo junta otro/fecha_hora/comandos/frases de actuador).
    y_activacion_completa = etiquetas_activacion(y_fina)
    limite_act = min(int((y_activacion_completa == k).sum()) for k in range(len(ACTIVACION_CLASES)))
    idx_bal_act = balancear_submuestreo(y_activacion_completa, limite_act, SEMILLA)
    datos_act = [datos[i] for i in idx_bal_act]
    X_act = X[idx_bal_act]
    y_activacion = y_activacion_completa[idx_bal_act]
    print("\nActivacion balanceada: %d de %d muestras." % (len(idx_bal_act), len(datos)))

    modelo_activacion, m_act, p_act_cv = entrenar_grupo("activacion", datos_act, X_act, y_activacion, ACTIVACION_CLASES, rapido)

    idx_si = ACTIVACION_CLASES.index("activacion")
    print("\nUmbral de activacion (validacion cruzada del ganador de 'activacion'):")
    tabla = tabla_umbrales_clase(y_activacion, p_act_cv, idx_si)
    for u, tpr, fpr in tabla:
        print("   umbral %.1f -> detecta %3.0f%% de las frases, %3.0f%% de falsos positivos" % (u, 100 * tpr, 100 * fpr))
    umbral_act = max(tabla, key=lambda t: t[1] - t[2])[0]
    print("Umbral sugerido: %.1f  -> config.UMBRAL_CONFIANZA" % umbral_act)

    # --- 2) Intencion: otro/fecha_hora/comandos, solo con muestras que no son
    # de actuador (esas ya no participan de esta decision, ver paso 3).
    mask_no_actuador = (y_fina < IDX_FINA_ACTUADOR_INI) | (y_fina >= IDX_FINA_ACTUADOR_FIN)
    datos_pre_int = [d for d, es in zip(datos, mask_no_actuador) if es]
    X_pre_int = X[mask_no_actuador]
    y_intencion_completa = etiquetas_intencion(y_fina[mask_no_actuador])
    limite_int = min(int((y_intencion_completa == k).sum()) for k in range(len(INTENCION_CLASES)))
    idx_bal_int = balancear_submuestreo(y_intencion_completa, limite_int, SEMILLA)
    datos_int = [datos_pre_int[i] for i in idx_bal_int]
    X_int = X_pre_int[idx_bal_int]
    y_intencion = y_intencion_completa[idx_bal_int]
    print("\nIntencion balanceada: %d de %d muestras." % (len(idx_bal_int), len(datos_pre_int)))

    modelo_intencion, m_int, p_int_cv = entrenar_grupo("intencion", datos_int, X_int, y_intencion, INTENCION_CLASES, rapido)

    idx_actr = INTENCION_CLASES.index("comandos")
    print("\nUmbral de la palabra 'comandos' (validacion cruzada del ganador de intencion):")
    tabla_actr = tabla_umbrales_clase(y_intencion, p_int_cv, idx_actr)
    for u, tpr, fpr in tabla_actr:
        print("   umbral %.1f -> detecta %3.0f%% de los casos, %3.0f%% de falsos positivos" % (u, 100 * tpr, 100 * fpr))
    umbral_actr_palabra = max(tabla_actr, key=lambda t: t[1] - t[2])[0]
    print("Umbral sugerido: %.1f  -> config.UMBRAL_PREGUNTA" % umbral_actr_palabra)

    idx_fecha = INTENCION_CLASES.index("fecha_hora")
    print("\nUmbral de 'fecha_hora' (validacion cruzada del ganador de intencion, referencia):")
    for u, tpr, fpr in tabla_umbrales_clase(y_intencion, p_int_cv, idx_fecha):
        print("   umbral %.1f -> detecta %3.0f%% de los casos, %3.0f%% de falsos positivos" % (u, 100 * tpr, 100 * fpr))

    # --- 3) Actuador: 4 clases directas (antes equipo+verbo encadenados), solo
    # con las frases de actuador de bombillo/secador (motobomba/maquinilla
    # pausados quedan fuera porque no estan en ACTUADOR_CLASES).
    mask_actuador = np.array([IDX_FINA_ACTUADOR_INI <= i < IDX_FINA_ACTUADOR_FIN and FINE_CLASES[i] in ACTUADOR_CLASES
                              for i in y_fina])
    datos_pre_actr = [d for d, es in zip(datos, mask_actuador) if es]
    X_pre_actr = X[mask_actuador]
    y_actuador_completa = etiquetas_actuador(y_fina[mask_actuador])
    limite_actr = min(int((y_actuador_completa == k).sum()) for k in range(len(ACTUADOR_CLASES)))
    idx_bal_actr = balancear_submuestreo(y_actuador_completa, limite_actr, SEMILLA)
    datos_actr = [datos_pre_actr[i] for i in idx_bal_actr]
    X_actr = X_pre_actr[idx_bal_actr]
    y_actuador = y_actuador_completa[idx_bal_actr]
    print("\nActuador balanceado: %d de %d muestras." % (len(idx_bal_actr), len(datos_pre_actr)))

    modelo_actuador, m_actr, p_actr_cv = entrenar_grupo("actuador", datos_actr, X_actr, y_actuador, ACTUADOR_CLASES, rapido)

    print("\nConfianza del actuador (validacion cruzada del ganador de 'actuador'):")
    pred_actr = p_actr_cv.argmax(axis=1)
    conf_actr = p_actr_cv.max(axis=1)
    acierto_actr = pred_actr == y_actuador
    for u in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        acepta = conf_actr >= u
        pct_correctos = 100 * acierto_actr[acepta].mean() if acepta.any() else float("nan")
        print("   umbral %.1f -> acepta %3.0f%% de los casos, %3.0f%% de esos son correctos" % (
            u, 100 * acepta.mean(), pct_correctos))
    print("Ajustar config.UMBRAL_ACTUADOR a mano con esta tabla (sin clase 'otro' aqui: un umbral")
    print("muy bajo puede ejecutar un actuador con un sonido que no era ninguno de los 4 comandos).")

    # --- 4) Persona y genero: quien habla, usando el tono y el centroide espectral
    # (ver espectro.py; corre una sola vez por frase, no por bloque) ademas de
    # envolvente/zcr/duracion. Solo con muestras de personas conocidas (hablante
    # poblado por preparar_dataset_pc.py), que sean voz real (no "otro": el
    # ruido/aplausos tambien queda con un hablante por el nombre del archivo,
    # pero no es su voz) y donde se pudo medir el tono (f0 > 0). El centroide no
    # se exige (>0) por separado: si el tono se pudo medir, el centroide tambien
    # (pide menos muestras), asi que no reduce mas el conjunto; si faltara, se
    # usa 0.0 como valor neutro.
    hablantes = [d.get("hablante") for d in datos]
    f0s = np.array([d.get("f0", 0.0) for d in datos])
    centroides = np.array([d.get("centroide", 0.0) for d in datos])
    idx_fina_otro = FINE_CLASES.index("otro")
    mask_id = np.array([h in PERSONA_CLASES and f > 0 and fi != idx_fina_otro
                         for h, f, fi in zip(hablantes, f0s, y_fina)])
    datos_id = [d for d, es in zip(datos, mask_id) if es]
    extra_id = np.stack([f0s[mask_id], centroides[mask_id]], axis=1)
    X_id = np.concatenate([X[mask_id], extra_id], axis=1)
    y_persona = np.array([PERSONA_CLASES.index(d["hablante"]) for d in datos_id])
    y_genero = np.array([GENERO_CLASES.index(GENERO_DE_PERSONA[d["hablante"]]) for d in datos_id])
    print("\nMuestras con hablante conocido y tono detectado: %d de %d" % (len(datos_id), len(datos)))

    modelo_persona, m_per, _ = entrenar_grupo("persona", datos_id, X_id, y_persona, PERSONA_CLASES, rapido, extra_id)
    modelo_genero, m_gen, _ = entrenar_grupo("genero", datos_id, X_id, y_genero, GENERO_CLASES, rapido, extra_id)

    exportar_cascada("modelo_pesos.py", {
        "activacion": (modelo_activacion, ACTIVACION_CLASES),
        "intencion": (modelo_intencion, INTENCION_CLASES),
        "actuador": (modelo_actuador, ACTUADOR_CLASES),
        "persona": (modelo_persona, PERSONA_CLASES),
        "genero": (modelo_genero, GENERO_CLASES),
    })
    print("\nExportado a modelo_pesos.py (cascada).")
    print("  activacion acc=%.3f | intencion acc=%.3f | actuador acc=%.3f | persona acc=%.3f | genero acc=%.3f" % (
        m_act["acc"], m_int["acc"], m_actr["acc"], m_per["acc"], m_gen["acc"]))


if __name__ == "__main__":
    main()
