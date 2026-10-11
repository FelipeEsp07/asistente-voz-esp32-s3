"""Arma la respuesta de hora y fecha como una lista de clips de clips/.

Cada palabra es un archivo clips/<palabra>.wav (preparar_clips.py). La
respuesta se reproduce en orden, sin librerias externas. Ejemplos:

    "fecha_hora"    -> hoy estamos a cuatro de octubre y son las siete y media de la tarde
                       (dices "fecha y hora")

Reglas de la hora: se dice la hora siguiente cuando faltan 31 minutos o mas
("menos cuarto", "menos veinte"), y "y cuarto", "y media" o "y <minutos>" en
el resto. La hora 1 se dice "es la una".
"""

UNIDADES = ("cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve")
ESPECIALES = {
    10: "diez", 11: "once", 12: "doce", 13: "trece", 14: "catorce", 15: "quince",
    16: "dieciseis", 17: "diecisiete", 18: "dieciocho", 19: "diecinueve",
    20: "veinte", 21: "veintiuno", 22: "veintidos", 23: "veintitres", 24: "veinticuatro",
    25: "veinticinco", 26: "veintiseis", 27: "veintisiete", 28: "veintiocho", 29: "veintinueve",
    30: "treinta", 40: "cuarenta", 50: "cincuenta",
}
MESES = ("enero", "febrero", "marzo", "abril", "mayo", "junio",
         "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre")
CARPETA = "clips"
# Cada palabra se guarda como clips/<persona>_<palabra>.wav, para poder responder
# con la voz de quien hablo (pendiente: identificar quien habla y elegir persona).
PERSONA_POR_DEFECTO = "felipe"


def _numero(n):
    """Palabras para un numero de 0 a 59."""
    if n < 10:
        return [UNIDADES[n]]
    if n in ESPECIALES:
        return [ESPECIALES[n]]
    decena = (n // 10) * 10
    return [ESPECIALES[decena], "y", UNIDADES[n % 10]]


def _periodo(hora24):
    if 5 <= hora24 <= 11:
        return "de_la_manana"
    if 12 <= hora24 <= 19:
        return "de_la_tarde"
    return "de_la_noche"


def _palabras_hora(hora24, minuto):
    """Palabras de la hora (sin clips de ruta). hora24 de 0 a 23."""
    periodo = _periodo(hora24)  # la hora real decide el periodo: 11:45 es "de la manana"
    if minuto > 30:
        hora24 = (hora24 + 1) % 24
    h12 = hora24 % 12 or 12
    if h12 == 1:
        palabras = ["es_la", "una"]
    else:
        palabras = ["son_las"] + _numero(h12)

    if minuto == 0:
        pass
    elif minuto == 15:
        palabras += ["y", "cuarto"]
    elif minuto == 30:
        palabras += ["y", "media"]
    elif minuto < 30:
        palabras += ["y"] + _numero(minuto)
    elif minuto == 45:
        palabras += ["menos", "cuarto"]
    else:
        palabras += ["menos"] + _numero(60 - minuto)

    return palabras + [periodo]


def _palabras_fecha(dia, mes):
    return ["hoy_estamos_a"] + _numero(dia) + ["de", MESES[mes - 1]]


def secuencia(tipo, ahora, persona=PERSONA_POR_DEFECTO):
    """Lista de rutas de clips para responder, con la voz de `persona`.

    tipo: "fecha_hora" (fecha y hora).
    ahora: tupla de machine.RTC().datetime(): (anio, mes, dia, dia_semana, hora, minuto, segundo, subsegundo).
    persona: de quien son los clips de voz (clips/<persona>_<palabra>.wav).
    """
    _, mes, dia, _, hora, minuto = ahora[:6]
    palabras = _palabras_fecha(dia, mes) + ["y"] + _palabras_hora(hora, minuto)
    return [CARPETA + "/" + persona + "_" + p + ".wav" for p in palabras]
