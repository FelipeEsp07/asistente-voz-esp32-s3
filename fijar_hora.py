"""Fija el reloj interno de la placa con la hora de esta PC.

La placa no tiene reloj con bateria: al reiniciarse o perder energia vuelve a
2000. Ejecuta esto despues de cada reinicio, antes de main.ejecutar():

    python fijar_hora.py COM5

Requiere que la placa este en el REPL (no ejecutando el asistente).
"""

import datetime
import subprocess
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit("uso: python fijar_hora.py COM<n>")
    puerto = sys.argv[1]
    n = datetime.datetime.now()
    # MicroPython: (anio, mes, dia, dia_de_la_semana con lunes=0, hora, minuto, segundo, subsegundo)
    tupla = "({},{},{},{},{},{},{},0)".format(n.year, n.month, n.day, n.weekday(), n.hour, n.minute, n.second)
    comando = "import machine; machine.RTC().datetime({}); print('reloj:', machine.RTC().datetime())".format(tupla)
    subprocess.run([sys.executable, "-m", "mpremote", "connect", puerto, "exec", comando], check=True)


if __name__ == "__main__":
    main()
