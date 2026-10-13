"""Comandos de voz que encienden o apagan los actuadores.

ejecutar() recibe la etiqueta ya resuelta por el grupo "actuador" del modelo
(ver entrenar_modelo.py/inferencia.py), por ejemplo "activar_bombillo": no
clasifica nada aqui, solo aplica la accion.
"""

import config


def ejecutar(etiqueta, actuadores):
    """Aplica el comando. Devuelve False si la etiqueta no existe."""
    if etiqueta not in config.COMANDOS:
        return False
    nombre, accion = config.COMANDOS[etiqueta]
    if accion == "on":
        actuadores.encender(nombre)
    else:
        actuadores.apagar(nombre)
    return True
