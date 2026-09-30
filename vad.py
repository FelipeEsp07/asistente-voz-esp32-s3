"""Deteccion de actividad de voz (VAD) por umbral de energia.

Aisla una palabra/frase pronunciada como una lista de energias por bloque,
usando un piso de ruido adaptativo calculado durante los silencios.
"""

import config


class DetectorVoz:
    def __init__(self):
        self.piso_ruido = config.UMBRAL_RUIDO_INICIAL
        self._en_voz = False
        self._bloques_voz_consecutivos = 0
        self._bloques_silencio_consecutivos = 0
        self._segmento = []

    def _umbral(self):
        return self.piso_ruido * config.FACTOR_ACTIVACION

    def procesar_bloque(self, energia, carga=None):
        """Alimenta un bloque. La decision de voz/silencio usa `energia`.
        Devuelve un segmento (lista de `carga` de cada bloque, o de
        `energia` si no se pasa carga) cuando se completa una palabra
        aislada, o None si aun no hay nada que reportar."""

        elemento = energia if carga is None else carga
        es_voz = energia > self._umbral()

        if not self._en_voz:
            if es_voz:
                self._bloques_voz_consecutivos += 1
                self._segmento.append(elemento)
                if self._bloques_voz_consecutivos >= config.BLOQUES_MIN_VOZ:
                    self._en_voz = True
                    self._bloques_silencio_consecutivos = 0
            else:
                self._bloques_voz_consecutivos = 0
                self._segmento = []
                # actualiza el piso de ruido solo durante silencio confirmado
                self.piso_ruido = int(self.piso_ruido * 0.98 + energia * 0.02)
            return None

        # estamos dentro de una palabra en curso
        self._segmento.append(elemento)

        if es_voz:
            self._bloques_silencio_consecutivos = 0
        else:
            self._bloques_silencio_consecutivos += 1

        fin_por_silencio = self._bloques_silencio_consecutivos >= config.BLOQUES_MIN_SILENCIO
        fin_por_longitud = len(self._segmento) >= config.MAX_BLOQUES_PALABRA

        if fin_por_silencio or fin_por_longitud:
            segmento = self._segmento
            self._segmento = []
            self._en_voz = False
            self._bloques_voz_consecutivos = 0
            self._bloques_silencio_consecutivos = 0

            if len(segmento) < config.MIN_BLOQUES_PALABRA:
                return None  # demasiado corto: ruido/clic, se descarta
            return segmento

        return None
