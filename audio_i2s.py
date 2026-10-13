"""Inicializacion del bus I2S y lectura de bloques de audio del INMP441."""

import array
import math
import time
from machine import I2S, Pin

import config

# Tras iniciar el I2S, el microfono/circuito tarda unos segundos en
# asentarse: la energia arranca varios ordenes de magnitud por encima
# del nivel real y decae exponencialmente. Si no se descarta, arruina
# la calibracion del VAD. Medido empiricamente: ~5s son suficientes.
BLOQUES_DESCARTE_ARRANQUE = int(5 * config.SAMPLE_RATE / config.BLOCK_SAMPLES)


def iniciar_i2s():
    return I2S(
        0,
        sck=Pin(config.PIN_SCK),
        ws=Pin(config.PIN_WS),
        sd=Pin(config.PIN_SD),
        mode=I2S.RX,
        bits=32,
        format=I2S.MONO,
        rate=config.SAMPLE_RATE,
        ibuf=config.I2S_BUFFER_BYTES,
    )


class LectorAudio:
    """Lee bloques de muestras de 32 bits desde el INMP441 via I2S.

    Ademas mantiene, en paralelo, un buffer circular con las muestras
    crudas (ya recortadas a escala de 16 bits, igual que hace analizar()
    con `raw >> 16`) de los ultimos config.MAX_BLOQUES_PALABRA bloques
    leidos. No cambia nada de lo que ya devuelve leer_bloque() ni de lo
    que recibe feature_extractor.extraer(): sirve solo para poder
    reconstruir despues, bajo demanda, el audio crudo de una frase
    completa (ver muestras_de_ultimos_bloques), por ejemplo para
    tono.f0_mediana(). Usa array('h')/array('H') de tamano fijo,
    sobrescritos en el lugar, para no crecer sin limite en una placa con
    memoria acotada."""

    def __init__(self, i2s, descartar_arranque=True):
        self.i2s = i2s
        self._buf_bytes = bytearray(config.BLOCK_SAMPLES * 4)

        self._capacidad_buffer = config.MAX_BLOQUES_PALABRA
        self._muestras_por_bloque = config.BLOCK_SAMPLES
        # array('h'): entero con signo de 16 bits, misma escala que usa
        # analizar() y que preparar_dataset_pc.py sobre los WAV de la PC.
        self._crudas = array.array(
            "h", bytes(2 * self._capacidad_buffer * self._muestras_por_bloque)
        )
        # cuantas muestras valen realmente en cada bloque guardado (la
        # ultima lectura de un segmento puede venir parcial).
        self._longitudes = array.array("H", bytes(2 * self._capacidad_buffer))
        self._indice_siguiente = 0  # proxima posicion del buffer circular a escribir

        if descartar_arranque:
            for _ in range(BLOQUES_DESCARTE_ARRANQUE):
                self.i2s.readinto(self._buf_bytes)

    def leer_bloque(self):
        """Devuelve un array('i', ...) con las muestras leidas (puede ser parcial).
        Tambien guarda una copia (en escala de 16 bits) en el buffer circular
        de muestras crudas, para muestras_de_ultimos_bloques()."""
        n_bytes = self.i2s.readinto(self._buf_bytes)
        n_muestras = n_bytes // 4
        muestras = array.array("i", self._buf_bytes[: n_muestras * 4])
        self._guardar_en_buffer_circular(muestras)
        return muestras

    def _guardar_en_buffer_circular(self, muestras):
        """Copia `muestras` (escala de 16 bits via >> 16) en la siguiente
        posicion del buffer circular, sobrescribiendo la entrada mas
        vieja. No asigna memoria nueva: todo el espacio ya esta reservado
        desde __init__."""
        indice = self._indice_siguiente
        base = indice * self._muestras_por_bloque
        n = len(muestras)
        if n > self._muestras_por_bloque:
            n = self._muestras_por_bloque
        for i in range(n):
            self._crudas[base + i] = muestras[i] >> 16
        self._longitudes[indice] = n
        self._indice_siguiente = (indice + 1) % self._capacidad_buffer

    def muestras_de_ultimos_bloques(self, cantidad_bloques):
        """Devuelve un array('h', ...) con las muestras crudas de los
        ultimos `cantidad_bloques` bloques leidos (los mas recientes),
        concatenadas en orden cronologico.

        Pensado para llamarse justo cuando vad.DetectorVoz cierra un
        segmento de `len(segmento)` bloques: como los bloques de un
        segmento son siempre los ultimos bloques leidos (son contiguos y
        se acaban de cerrar), pasando cantidad_bloques=len(segmento) se
        reconstruye el audio crudo de esa frase completa, por ejemplo para
        tono.f0_mediana(). Si se pide mas de la capacidad del buffer
        (config.MAX_BLOQUES_PALABRA), se recorta a lo que hay disponible."""
        if cantidad_bloques > self._capacidad_buffer:
            cantidad_bloques = self._capacidad_buffer
        if cantidad_bloques <= 0:
            return array.array("h")

        primero = (self._indice_siguiente - cantidad_bloques) % self._capacidad_buffer
        total_muestras = 0
        for paso in range(cantidad_bloques):
            indice = (primero + paso) % self._capacidad_buffer
            total_muestras += self._longitudes[indice]

        salida = array.array("h", bytes(2 * total_muestras))
        pos = 0
        for paso in range(cantidad_bloques):
            indice = (primero + paso) % self._capacidad_buffer
            n = self._longitudes[indice]
            base = indice * self._muestras_por_bloque
            for i in range(n):
                salida[pos + i] = self._crudas[base + i]
            pos += n
        return salida

    def vaciar_buffer(self):
        """Descarta el audio acumulado en el buffer DMA mientras no se
        estaba leyendo (por ejemplo, mientras sonaba el parlante). Sin
        esto, la siguiente lectura trae audio viejo y el VAD puede
        confundir el eco del propio asistente con una orden nueva."""
        for _ in range(30):
            t0 = time.ticks_us()
            self.leer_bloque()
            if time.ticks_diff(time.ticks_us(), t0) > 15000:
                break

    def analizar(self, muestras):
        """Una sola pasada por el bloque. Devuelve (energia, carga):
        - energia: media de muestras^2 en escala de 32 bits, sin offset DC
          (es lo que usa el VAD).
        - carga: (rms16, hf, zcr) con
            rms16: RMS en escala de 16 bits,
            hf:    energia de la primera diferencia / energia total
                   (cuanto mas alto, mas contenido de altas frecuencias),
            zcr:   fraccion de cruces por cero.
        Se trabaja con muestras de 16 bits (>> 16): son enteros chicos de
        MicroPython, mucho mas rapidos que los enteros grandes de 32 bits.
        Se resta la media del bloque: el INMP441 entrega un offset de
        continua (~-200) que deriva entre bloques y, sin restarlo, domina
        la energia en silencio, ensucia el VAD y anula zcr/hf."""
        n = len(muestras)
        if n < 2:
            return 0, (0.0, 0.0, 0.0)

        dc = (sum(muestras) // n) >> 16
        suma_e = 0
        suma_d = 0
        cruces = 0
        prev = (muestras[0] >> 16) - dc
        for raw in muestras:
            s = (raw >> 16) - dc
            d = s - prev
            suma_e += s * s
            suma_d += d * d
            if (s ^ prev) < 0:
                cruces += 1
            prev = s

        energia = (suma_e << 32) // n
        rms16 = math.sqrt(suma_e / n)
        hf = suma_d / suma_e if suma_e > 0 else 0.0
        zcr = cruces / (n - 1)
        return energia, (round(rms16, 1), round(hf, 3), round(zcr, 3))
