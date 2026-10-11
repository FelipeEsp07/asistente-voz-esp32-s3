"""Comandos de voz que encienden o apagan los actuadores.

clasificar() esta pendiente: necesita un modelo entrenado con grabaciones de
los comandos. Mientras tanto devuelve None y el asistente responde que no
entendio, sin inventar acciones.
"""

import config


def clasificar(caracteristicas):
    """Etiqueta del comando dicho, o None si todavia no hay clasificador."""
    return None


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
