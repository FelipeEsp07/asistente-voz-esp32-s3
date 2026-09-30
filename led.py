"""Control del LED indicador de activacion (GPIO simple, sin PWM)."""

from machine import Pin

import config

_pin = Pin(config.PIN_LED, Pin.OUT)
_pin.value(0)


def encender():
    _pin.value(1)


def apagar():
    _pin.value(0)
