# Guia de grabaciones

Esta guia lista todo lo que hay que grabar para entrenar el asistente y responder
con la voz de cada persona. Grabar **en la computadora**, no con el microfono del
ESP32 (el INMP441 esta danado y no capta bien).

## Como grabar

Se graba con el microfono de la PC usando `grabar_wav.py`:

```
python grabar_wav.py felipe hola_pan 10
python grabar_wav.py laura comando_activar_secador 5 --segundos 2.5
```

- Pulsa Enter, di la frase y espera a que termine la toma (2,5 s por defecto).
- Si la toma sale saturada o casi en silencio, el programa avisa y pide repetirla.
- Nunca sobrescribe: si ya hay tomas de esa persona y grupo, continua la numeracion.
- Si el microfono correcto no es el predeterminado: `python grabar_wav.py --listar-dispositivos`
  y luego agrega `--dispositivo <numero>`.

Resultado: `grabaciones/<grupo>/<persona>_<grupo>_<numero>.wav` (16 kHz, mono, 16 bits).

- **Lugar**: cuarto silencioso, sin ventilador, sin musica, sin TV de fondo.
- **Microfono**: el mismo para todas las personas, idealmente. Mantener la misma
  distancia durante una serie.
- **Duracion**: 1 a 3 segundos por toma. Hablar en tono natural, sin gritar ni susurrar.
- **Nombre de archivo**: lo asigna el programa. Ejemplo: `felipe_hola_pan_07.wav`,
  `laura_comando_activar_secador_03.wav`.
- **Carpeta**: una carpeta por grupo (`grabaciones/hola_pan/`, `grabaciones/comando_activar_secador/`, etc.).
- **Variar**: en cada serie, cambiar un poco el volumen, la velocidad y la distancia.

## 1. Frase de activacion: "Hola PAN"

Cada una de las 4 personas (Felipe y 3 companeros):

| Que decir | Cuantas veces |
|---|---|
| "Hola PAN" | 30 (10 cerca, 10 a un metro, 10 a dos metros o con otro volumen) |

Ademas, cada persona dice **otras frases** que no son "Hola PAN" (son las negativas):

| Que decir | Cuantas veces |
|---|---|
| "Hola" (sola) | 5 |
| "Hola, como estas" | 5 |
| "PAN" (sola) | 5 |
| "Activar asistente" (la frase vieja) | 5 |
| Cualquier frase de 2 o 3 palabras del mismo tipo | 10 |

Total por persona: 30 de "Hola PAN" + 30 de otras frases = **60 tomas**.

## 2. Comandos para encender los actuadores

Primero se dice la frase de activacion, luego el comando.

| Comando | Cuantas veces por persona |
|---|---|
| "Activar cautin" | 5 |
| "Activar secador" | 5 |
| "Activar ventilador" | 5 |
| "Activar [cuarto actuador]" | 5 *(pendiente: falta su nombre)* |

## 3. Comandos para apagar los actuadores

| Comando | Cuantas veces por persona |
|---|---|
| "Apagar cautin" | 5 |
| "Apagar secador" | 5 |
| "Apagar ventilador" | 5 |
| "Apagar [cuarto actuador]" | 5 *(pendiente)* |

Total de comandos por persona: 8 comandos x 5 = **40 tomas** (48 cuando se confirme el cuarto actuador).

## 4. Respuestas de confirmacion (en tu propia voz)

Cada persona graba una sola vez estas frases. Son las que responde el asistente.

| Frase | Cuando se usa |
|---|---|
| "Te escucho" (o "Si, dime") | Al reconocer "Hola PAN" |
| "Listo" | Despues de ejecutar un comando |
| "No entendi, repite" | Cuando no reconoce el comando (opcional por ahora) |

## 5. Ruido y ambiente (sin persona)

Estas tomas enseñan al modelo lo que **no** es voz. Grabar en el mismo cuarto, unas 20 en total:

- Silencio del cuarto (5 tomas de 3 s)
- Ventilador o aire acondicionado encendido (3 tomas)
- Teclado, mouse, golpes suaves en la mesa (4 tomas)
- Aplausos y chasquidos de dedos (3 tomas)
- Tos y estornudo (2 tomas)
- Musica o TV de fondo a volumen normal (3 tomas)

## 6. Vocabulario para decir la hora y la fecha

Cada persona graba **una vez** cada palabra. Estas piezas se unen para armar frases como
"son las tres y veinte" o "hoy es lunes, uno de octubre de dos mil veintiseis".

**Numeros** (una toma cada uno):
- cero, uno, dos, tres, cuatro, cinco, seis, siete, ocho, nueve
- diez, once, doce, trece, catorce, quince, dieciseis, diecisiete, dieciocho, diecinueve
- veinte, veintiuno, veintidos, veintitres, veinticuatro, veinticinco, veintiseis, veintisiete, veintiocho, veintinueve
- treinta, cuarenta, cincuenta
- Nota: para 31 a 59 el sistema junta "treinta" + "y" + "uno", etc.

**Conectores**:
- "son las", "es la", "y", "menos", "en punto", "cuarto", "media"
- "hoy es", "de", "del", "dos mil"

**Dias de la semana** (una toma cada uno):
- lunes, martes, miercoles, jueves, viernes, sabado, domingo

**Meses** (una toma cada uno):
- enero, febrero, marzo, abril, mayo, junio, julio, agosto, septiembre, octubre, noviembre, diciembre

Total por persona: unas 70 tomas. Se pueden grabar en una sola sesion, diciendo cada palabra
en voz clara, separadas por medio segundo de silencio.

## Resumen de cantidades

| Grupo | Por persona | Total (4 personas) |
|---|---|---|
| "Hola PAN" | 30 | 120 |
| Otras frases (negativas de persona) | 30 | 120 |
| Comandos activar/apagar (8, o 10 con el cuarto actuador) | 40 (o 50) | 160 (o 200) |
| Respuestas de confirmacion | 3 | 12 |
| Vocabulario de hora y fecha | 70 | 280 |
| Ruido y ambiente (sin persona, una sola vez) | 20 | 20 |

## 7. Preguntas de hora y fecha (grabadas con `--preguntas`)

Son las frases que dices para pedir la hora o la fecha. 5 tomas por frase:
"qué hora es", "qué hora es ahora", "dime la hora", "qué día es hoy", "qué fecha es hoy",
"fecha". Se graban 15 tomas de cada una.

```
python grabar_wav.py felipe --preguntas
```

Se guardan en `grabaciones/preguntas_hora_fecha/`. Estas tomas entrenan la detección de la
intención; la respuesta sale de las palabras del vocabulario (sección 6).

Ejemplo de respuesta a "fecha y hora":
"hoy estamos a 4 de octubre y son las 7:30 de la tarde".

Las palabras de la respuesta que faltaban en la sección 6 se agregaron con
`python grabar_wav.py felipe --vocabulario`: "hoy estamos a", "de la mañana", "de la tarde", "de la noche".

## Pendiente

- Nombre del cuarto actuador (en la lista aparece como "[cuarto actuador]").
- Nombre y genero de los tres companeros, para ajustar la respuesta de voces desconocidas.
