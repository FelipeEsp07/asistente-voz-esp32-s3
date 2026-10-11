"""Punto de entrada del asistente.

MicroPython ejecuta main.py al arrancar la placa. Este archivo no inicia nada
al encender, para que el REPL quede libre. Para iniciar el asistente:

    >>> import main
    >>> main.ejecutar()
"""

from asistente import ejecutar
