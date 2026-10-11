"""Control de actuadores (reles) con apagado de seguridad.

Reglas:
- Todo actuador arranca apagado.
- Cada encendido tiene un tiempo maximo y se apaga solo al cumplirlo, aunque
  el resto del programa falle o se quede esperando.
- Solo este modulo escribe en los pines de los reles.

crear_pin y reloj_ms se pasan como parametros: en la placa son machine.Pin y
time.ticks_ms; en la PC se pueden simular para probar sin hardware.
"""

MASCARA_TICKS = 0x3FFFFFFF  # time.ticks_ms de MicroPython da la vuelta en 2**30

# El modulo de reles de 4 canales se activa con nivel BAJO (comprobado en placa).
NIVEL_ENCENDIDO = 0
NIVEL_APAGADO = 1


class Actuadores:
    def __init__(self, configuracion, crear_pin, reloj_ms):
        """configuracion: {nombre: (numero_de_pin, segundos_maximos_encendido)}."""
        self._reloj_ms = reloj_ms
        self._pines = {}
        self._segundos_max = {}
        self._inicio_ms = {}
        for nombre, (numero_pin, segundos_max) in configuracion.items():
            pin = crear_pin(numero_pin)
            pin.value(NIVEL_APAGADO)  # arranca apagado: el rele no se activa al encender la placa
            self._pines[nombre] = pin
            self._segundos_max[nombre] = segundos_max
            self._inicio_ms[nombre] = None

    def encender(self, nombre):
        self._pines[nombre].value(NIVEL_ENCENDIDO)
        self._inicio_ms[nombre] = self._reloj_ms()

    def apagar(self, nombre):
        self._pines[nombre].value(NIVEL_APAGADO)
        self._inicio_ms[nombre] = None

    def encendido(self, nombre):
        return self._inicio_ms[nombre] is not None

    def revisar(self):
        """Apaga los actuadores que superaron su tiempo maximo. Llamar en cada ciclo."""
        ahora = self._reloj_ms()
        for nombre, inicio in self._inicio_ms.items():
            if inicio is None:
                continue
            transcurrido_s = ((ahora - inicio) & MASCARA_TICKS) / 1000
            if transcurrido_s >= self._segundos_max[nombre]:
                self.apagar(nombre)

    def apagar_todos(self):
        for nombre in self._pines:
            self.apagar(nombre)
