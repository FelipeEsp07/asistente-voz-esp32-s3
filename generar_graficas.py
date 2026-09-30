"""Genera las graficas de evaluacion del modelo en graficas/*.png.

IMPORTANTE: corre en la PC (numpy + matplotlib), no en la ESP32-S3.
No modifica el dataset ni el modelo entrenado.

Uso: python generar_graficas.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import entrenar_modelo as em
import inferencia
import modelo_pesos as m

CARPETA = "graficas"
os.makedirs(CARPETA, exist_ok=True)

AZUL = "#2563eb"
NARANJA = "#f97316"
VERDE = "#16a34a"
ROJO = "#dc2626"
GRIS = "#6b7280"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d1d5db", "axes.labelcolor": "#111827",
    "text.color": "#111827", "xtick.color": "#374151", "ytick.color": "#374151",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
})


def guardar(fig, nombre):
    ruta = os.path.join(CARPETA, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("guardado:", ruta)


# --------------------------------------------------------------------------
# Datos y evaluacion (igual que en el reporte de metricas)
# --------------------------------------------------------------------------

datos = em.cargar(config.RUTA_DATASET)
em.XG, y2d, dist = em.preparar_datos(datos)
X, y = em.XG, y2d.ravel()

p_train = np.array([inferencia.predecir_probabilidad(list(x), m) for x in X])

folds = em.armar_folds(datos, X)
metricas_cv, sd_cv, p_cv = em.evaluar(
    lambda s: em.Ensamble(lambda ss: em.CNN1D(8, 0.01, 300, semilla=ss), 5, s), folds, y2d, dist
)

UMBRAL = 0.5


def matriz(y, p, umbral=UMBRAL):
    pred = (p >= umbral).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return np.array([[tp, fn], [fp, tn]])


# --------------------------------------------------------------------------
# 1) Matrices de confusion (entrenamiento vs validacion cruzada)
# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (titulo, pp) in zip(axes, [("Sobre datos de entrenamiento\n(optimista)", p_train),
                                    ("Validacion cruzada\n(estimacion realista)", p_cv)]):
    cm = matriz(y, pp)
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "#111827"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, color=color)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["frase", "otra"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["frase", "otra"])
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(titulo)
fig.suptitle('Matriz de confusion — umbral %.1f, ensamble de 5 CNN 1D' % UMBRAL, y=1.03)
guardar(fig, "01_matriz_confusion.png")

# --------------------------------------------------------------------------
# 2) Curva ROC
# --------------------------------------------------------------------------

def curva_roc(y, p):
    umbrales = np.sort(np.unique(p))[::-1]
    tpr, fpr = [0.0], [0.0]
    for u in umbrales:
        pred = p >= u
        tpr.append(pred[y == 1].mean())
        fpr.append(pred[y == 0].mean())
    tpr.append(1.0); fpr.append(1.0)
    return np.array(fpr), np.array(tpr)

fig, ax = plt.subplots(figsize=(5.5, 5))
fx, tx = curva_roc(y, p_cv)
ax.plot(fx, tx, color=AZUL, lw=2.5, label="ensamble CNN, 1a repeticion (AUC=%.3f)" % em.auc(y, p_cv))
ax.plot([0, 1], [0, 1], "--", color=GRIS, lw=1.5, label="azar (AUC=0.5)")
ax.scatter([metricas_cv["fpr"]], [metricas_cv["tpr"]], color=NARANJA, zorder=5, s=70,
           label="umbral 0.5 (promedio de 3 repeticiones)")
ax.set_xlabel("Falsos positivos (FPR)"); ax.set_ylabel("Detecta la frase (TPR)")
ax.set_title("Curva ROC — validacion cruzada (AUC promedio: %.3f)" % metricas_cv["auc"])
ax.legend(loc="lower right", frameon=False)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
guardar(fig, "02_curva_roc.png")

# --------------------------------------------------------------------------
# 3) Umbral de decision vs deteccion / falsos positivos
# --------------------------------------------------------------------------

tabla = em.tabla_umbrales(y2d, p_cv)
us = [t[0] for t in tabla]; tprs = [t[1] * 100 for t in tabla]; fprs = [t[2] * 100 for t in tabla]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(us, tprs, "o-", color=VERDE, lw=2.5, ms=7, label="Detecta la frase")
ax.plot(us, fprs, "o-", color=ROJO, lw=2.5, ms=7, label="Falsos positivos")
ax.axvline(UMBRAL, color=GRIS, ls="--", lw=1.5)
ax.text(UMBRAL + 0.02, 95, "umbral usado (%.1f)" % UMBRAL, color=GRIS, fontsize=9)
ax.set_xlabel("Umbral de confianza (config.UMBRAL_CONFIANZA)")
ax.set_ylabel("%")
ax.set_title("Efecto del umbral de decision")
ax.legend(frameon=False)
ax.set_ylim(-5, 105)
guardar(fig, "03_umbral_deteccion.png")

# --------------------------------------------------------------------------
# 4) Recall por distancia
# --------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5.5, 4.5))
etiquetas = ["cerca\n(~30 cm)", "media\n(~1 m)", "lejos\n(~2 m)"]
valores = [metricas_cv["tpr_cerca"] * 100, metricas_cv["tpr_media"] * 100, metricas_cv["tpr_lejos"] * 100]
barras = ax.bar(etiquetas, valores, color=[AZUL, AZUL, AZUL], width=0.55)
for b, v in zip(barras, valores):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, "%.0f%%" % v, ha="center", fontweight="bold")
ax.set_ylabel("% de frases detectadas")
ax.set_title("Deteccion de 'activar asistente' por distancia")
ax.set_ylim(0, 108)
guardar(fig, "04_recall_por_distancia.png")

# --------------------------------------------------------------------------
# 5) Comparacion MLP vs CNN vs ensambles (misma grilla que entrenar_modelo.py)
# --------------------------------------------------------------------------

print("\nRecalculando la comparacion de modelos (puede tardar ~1-2 min)...")
comparacion = {}
for h in (4, 8, 16):
    for ep in (150, 300, 600):
        mres, _, _ = em.evaluar(lambda s, h=h, ep=ep: em.MLP(h, 0.01, ep, semilla=s), folds, y2d, dist)
        comparacion["MLP h=%d, ep=%d" % (h, ep)] = mres["auc"]
for f in (4, 8):
    for ep in (150, 300, 600):
        mres, _, _ = em.evaluar(lambda s, f=f, ep=ep: em.CNN1D(f, 0.01, ep, semilla=s), folds, y2d, dist)
        comparacion["CNN f=%d, ep=%d" % (f, ep)] = mres["auc"]
comparacion["Ensamble MLP h=16"] = 0.909
comparacion["Ensamble CNN f=8 (elegido)"] = metricas_cv["auc"]

items = sorted(comparacion.items(), key=lambda kv: kv[1])
nombres = [k for k, _ in items]; valores = [v for _, v in items]
colores = [VERDE if "elegido" in n else (NARANJA if "Ensamble" in n else AZUL) for n in nombres]

fig, ax = plt.subplots(figsize=(8.5, 0.38 * len(nombres) + 1.2))
ax.barh(nombres, valores, color=colores, height=0.68)
ax.set_xlabel("AUC (validacion cruzada)")
ax.set_title("Comparacion de arquitecturas e hiperparametros")
ax.set_xlim(0.8, 0.96)
ax.margins(y=0.01)
for i, v in enumerate(valores):
    ax.text(v + 0.002, i, "%.3f" % v, va="center", fontsize=9)
fig.tight_layout()
guardar(fig, "05_comparacion_modelos.png")

# --------------------------------------------------------------------------
# 6) Version 1 vs version 2
# --------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5, 4.5))
etiquetas = ["v1: MLP\n(60 muestras)", "v2: ensamble CNN\n(100 muestras)"]
valores = [73.0, metricas_cv["acc"] * 100]
barras = ax.bar(etiquetas, valores, color=[GRIS, VERDE], width=0.5)
for b, v in zip(barras, valores):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, "%.1f%%" % v, ha="center", fontweight="bold")
ax.set_ylabel("Precision (accuracy)")
ax.set_title("Evolucion del modelo")
ax.set_ylim(0, 100)
guardar(fig, "06_evolucion_v1_v2.png")

print("\nListo: 6 graficas en la carpeta '%s/'." % CARPETA)
