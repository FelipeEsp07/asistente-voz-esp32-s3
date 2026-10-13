"""Copia a la PC las tomas grabadas con el microfono de la placa.

En el REPL de la placa, graba las tomas con grabar_placa.py:
    >>> import grabar_placa
    >>> grabar_placa.tomas("fecha", 30)

Luego, desde la PC:
    python copiar_grabaciones_placa.py COM5 fecha felipe

Copia /fecha_NN.wav de la placa a grabaciones/preguntas_placa/felipe_fecha_NN.wav,
el nombre que espera preparar_dataset_pc.py --preguntas.
"""

import os
import re
import subprocess
import sys

CARPETA_DESTINO = os.path.join("grabaciones", "preguntas_placa")


def _archivos_en_placa(puerto, frase):
    salida = subprocess.run([sys.executable, "-m", "mpremote", "connect", puerto, "fs", "ls", ":"],
                            check=True, capture_output=True, text=True).stdout
    patron = re.compile(r"{}_(\d+)\.wav$".format(re.escape(frase)))
    numeros = []
    for linea in salida.splitlines():
        coincidencia = patron.search(linea.strip())
        if coincidencia:
            numeros.append(coincidencia.group(1))
    return sorted(set(numeros))


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit("uso: python copiar_grabaciones_placa.py COM<n> <frase> <persona> [carpeta_destino]")
    puerto, frase, persona = sys.argv[1], sys.argv[2], sys.argv[3]
    carpeta = sys.argv[4] if len(sys.argv) == 5 else CARPETA_DESTINO

    os.makedirs(carpeta, exist_ok=True)
    numeros = _archivos_en_placa(puerto, frase)
    if not numeros:
        raise SystemExit("no hay tomas de '{}' en la placa".format(frase))

    for numero in numeros:
        origen = ":{}_{}.wav".format(frase, numero)
        destino = os.path.join(carpeta, "{}_{}_{}.wav".format(persona, frase, numero))
        subprocess.run([sys.executable, "-m", "mpremote", "connect", puerto, "cp", origen, destino], check=True)
        print("  copiado {} -> {}".format(origen, destino))
    print("{} tomas en {}/".format(len(numeros), carpeta))


if __name__ == "__main__":
    main()
