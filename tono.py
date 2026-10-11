"""Estimacion de la frecuencia fundamental (tono) de la voz por autocorrelacion.

Sirve para dos cosas de la arquitectura: el genero de quien habla (la voz
masculina suele estar entre ~85 y 180 Hz, la femenina entre ~165 y 255 Hz)
y como caracteristica para distinguir hablantes.

Corre en la placa y en la PC con el mismo codigo. Se usa una vez por frase,
no por bloque, por eso se diezma la señal a 4 kHz: el costo de la
autocorrelacion crece con el cuadrado de las muestras y los lags.
"""

DIEZMADO = 4          # 16 kHz -> 4 kHz
FS_TONO = 16000 // DIEZMADO
FMIN = 70
FMAX = 320
UMBRAL_VOZ = 0.45     # correlacion normalizada minima para considerar la trama sonora


def diezmar(x):
    """Promedio de DIEZMADO muestras: filtro pasa-bajos crudo, suficiente para el tono."""
    salida = []
    n = len(x) // DIEZMADO
    for i in range(n):
        s = 0
        base = i * DIEZMADO
        for k in range(DIEZMADO):
            s += x[base + k]
        salida.append(s / DIEZMADO)
    return salida


def _f0_trama(y, ini, n, lag_min, lag_max):
    e0 = 0.0
    for i in range(ini, ini + n):
        e0 += y[i] * y[i]
    if e0 <= 0:
        return 0.0, 0.0

    corr = {}
    for lag in range(lag_min - 1, lag_max + 2):
        r = 0.0
        for i in range(ini, ini + n - lag):
            r += y[i] * y[i + lag]
        corr[lag] = r / e0

    mejor_r = 0.0
    for lag in range(lag_min, lag_max + 1):
        if corr[lag] > mejor_r:
            mejor_r = corr[lag]
    if mejor_r < UMBRAL_VOZ:
        return 0.0, mejor_r

    # Error de octava: el maximo global puede ser un multiplo del periodo.
    # Se elige el periodo mas corto que sea pico local y casi tan alto como el maximo.
    mejor_lag = 0
    for lag in range(lag_min, lag_max + 1):
        es_pico = corr[lag] >= corr[lag - 1] and corr[lag] >= corr[lag + 1]
        if es_pico and corr[lag] >= 0.85 * mejor_r:
            mejor_lag = lag
            break
    if mejor_lag == 0:
        return 0.0, mejor_r

    r_prev = corr[mejor_lag - 1]
    r_actual = corr[mejor_lag]
    r_next = corr[mejor_lag + 1]

    denom = r_prev - 2 * r_actual + r_next
    desplaz = 0.5 * (r_prev - r_next) / denom if denom != 0 else 0.0
    return FS_TONO / (mejor_lag + desplaz), mejor_r


def f0_tramas(x, n_trama=512, salto=1024):
    """Devuelve la lista de frecuencias fundamentales por trama (0 si no es sonora).
    x: muestras de 16 kHz (enteros o flotantes)."""
    y = diezmar(x)
    lag_min = int(FS_TONO / FMAX)
    lag_max = int(FS_TONO / FMIN)
    n_trama_d = n_trama // DIEZMADO
    salto_d = salto // DIEZMADO
    f0s = []
    ini = 0
    while ini + n_trama_d + lag_max + 1 < len(y):
        f0, _ = _f0_trama(y, ini, n_trama_d, lag_min, lag_max)
        f0s.append(f0)
        ini += salto_d
    return f0s


def f0_mediana(x):
    """Tono representativo de una frase: mediana de las tramas sonoras. 0 si no hay voz."""
    sonoras = sorted(f for f in f0_tramas(x) if f > 0)
    if not sonoras:
        return 0.0
    return sonoras[len(sonoras) // 2]
