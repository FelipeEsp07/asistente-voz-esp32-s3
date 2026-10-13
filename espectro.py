"""Centroide espectral representativo de UNA FRASE COMPLETA, via FFT
(radix-2, puro Python/MicroPython, sin numpy). Corre UNA SOLA VEZ por frase
detectada, nunca por bloque: se decima, se parte en tramas y se toma la
mediana, igual patron que tono.f0_mediana. Da una caracteristica real de
timbre (mas fina que el cruce por cero), pensada para ayudar a "persona" a
distinguir voces con un tono parecido (ver entrenar_modelo.py).

Medido en la ESP32-S3: una FFT de 512 puntos (sin decimar) tarda ~139 ms --
demasiado para correr en cada bloque (~32 ms de presupuesto). Decimada por
4 (128 puntos, igual factor que tono.py) tarda ~22 ms, aceptable para unas
pocas tramas DESPUES de cerrar el segmento (ver MAX_TRAMAS).

Corre en la placa y en la PC con el MISMO codigo, para que las
caracteristicas de entrenamiento y de uso coincidan exactamente.
"""

import math

DIEZMADO = 4       # igual que tono.py: reduce el costo de la FFT
N = 128            # tramas de 512 muestras crudas, decimadas por 4
MAX_TRAMAS = 8     # tope de tramas evaluadas por frase (costo acotado:
                   # ~8 * 22 ms = 176 ms como mucho, sin importar cuan
                   # larga sea la frase)


def _log2_entero(n):
    """log2(n) para n potencia de 2, sin int.bit_length() (no esta en
    todas las variantes de MicroPython)."""
    bits = 0
    while n > 1:
        n >>= 1
        bits += 1
    return bits


_LOG2N = _log2_entero(N)
if (1 << _LOG2N) != N:
    raise ValueError("N debe ser potencia de 2")


def _bit_reversal(valor, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (valor & 1)
        valor >>= 1
    return r


_INDICES = [_bit_reversal(i, _LOG2N) for i in range(N)]
_COS = [math.cos(-2 * math.pi * k / N) for k in range(N // 2)]
_SEN = [math.sin(-2 * math.pi * k / N) for k in range(N // 2)]
_VENTANA = [0.5 - 0.5 * math.cos(2 * math.pi * i / (N - 1)) for i in range(N)]

_re = [0.0] * N
_im = [0.0] * N


def _fft():
    """FFT in-place (Cooley-Tukey, radix-2) sobre los buffers globales
    _re/_im, que ya deben tener la señal en orden bit-reversal."""
    tam = 2
    while tam <= N:
        mitad = tam // 2
        paso = N // tam
        for inicio in range(0, N, tam):
            for k in range(mitad):
                w_re = _COS[k * paso]
                w_sen = _SEN[k * paso]
                i_impar = inicio + k + mitad
                i_par = inicio + k
                tr = _re[i_impar] * w_re - _im[i_impar] * w_sen
                ti = _re[i_impar] * w_sen + _im[i_impar] * w_re
                _re[i_impar] = _re[i_par] - tr
                _im[i_impar] = _im[i_par] - ti
                _re[i_par] += tr
                _im[i_par] += ti
        tam *= 2


def _centroide_trama(trama):
    """`trama`: N muestras (ya decimadas). Centroide espectral normalizado
    a [0, 1] (0 = toda la energia cerca de continua/graves, 1 = toda la
    energia en el bin mas agudo representable). 0.0 si no hay energia."""
    for i in range(N):
        idx = _INDICES[i]
        _re[i] = trama[idx] * _VENTANA[idx]
        _im[i] = 0.0
    _fft()
    num = 0.0
    den = 0.0
    mitad = N // 2
    for k in range(1, mitad):  # se excluye k=0 (continua)
        mag = math.sqrt(_re[k] * _re[k] + _im[k] * _im[k])
        num += k * mag
        den += mag
    return (num / den) / mitad if den > 0.0 else 0.0


def _diezmar_ventana(x, inicio_crudo):
    """Decima SOLO una ventana de N*DIEZMADO muestras crudas a partir de
    `inicio_crudo` (no toda la frase): es lo unico que cada trama necesita,
    mucho mas barato que decimar el segmento completo de una (medido en la
    placa: decimar 24000 muestras completas cuesta ~900ms; decimar solo las
    512 que pide una trama, ~19ms)."""
    salida = [0.0] * N
    for i in range(N):
        s = 0.0
        base = inicio_crudo + i * DIEZMADO
        for k in range(DIEZMADO):
            s += x[base + k]
        salida[i] = s / DIEZMADO
    return salida


def centroide_frase(x):
    """Centroide espectral representativo de TODA la frase: evalua hasta
    MAX_TRAMAS ventanas repartidas a lo largo del segmento (decimando SOLO
    esas ventanas, no la frase completa) y devuelve la mediana de las que
    tienen energia, normalizada a [0, 1]. 0.0 si el segmento es muy corto o
    no tiene energia.

    `x`: muestras crudas de 16 kHz del segmento completo (por ejemplo,
    lector.muestras_de_ultimos_bloques() en la placa, o el recorte
    x[ini*B:fin*B] en preparar_dataset_pc.py). Se llama UNA SOLA VEZ por
    frase detectada, nunca por bloque (ver docstring del modulo)."""
    ventana_cruda = N * DIEZMADO
    n_crudo = len(x)
    if n_crudo < ventana_cruda:
        return 0.0

    max_inicio = n_crudo - ventana_cruda
    cantidad = min(MAX_TRAMAS, max_inicio + 1)
    if cantidad <= 1:
        inicios = [0]
    else:
        paso = max_inicio / (cantidad - 1)
        inicios = [int(round(i * paso)) for i in range(cantidad)]

    valores = []
    for ini in inicios:
        trama = _diezmar_ventana(x, ini)
        c = _centroide_trama(trama)
        if c > 0.0:
            valores.append(c)
    if not valores:
        return 0.0
    valores.sort()
    return valores[len(valores) // 2]
