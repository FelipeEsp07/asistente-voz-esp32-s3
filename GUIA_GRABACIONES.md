# Guia de grabaciones

Esta guia lista todo lo que hay que grabar para entrenar el asistente y responder
con la voz de cada persona. Esta seccion usa el microfono de la PC por comodidad
para grabar en serie; para el dataset de entrenamiento real conviene el microfono
de la placa (ver la nota debajo: el INMP441 funciona bien cableado directo, sin
la breakout Freenove).

> Nota: para el dataset de entrenamiento real (el que usa `entrenar_modelo.py`
> para el modelo que corre en la placa) se prefiere grabar con el microfono de
> la placa (`grabar_placa.py` + `copiar_grabaciones_placa.py`, ver README), porque
> el microfono de la PC suena distinto al INMP441 y el modelo no generalizaba
> bien. Las instrucciones de esta guia (con `grabar_wav.py`, microfono de la PC)
> siguen sirviendo para el vocabulario de fecha y hora, las preguntas y para
> hacer pruebas rapidas.

## Como grabar

Se graba con el microfono de la PC usando `grabar_wav.py`:

```
python grabar_wav.py felipe hola_bender 10
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
- **Nombre de archivo**: lo asigna el programa. Ejemplo: `felipe_hola_bender_07.wav`,
  `laura_comando_activar_secador_03.wav`.
- **Carpeta**: una carpeta por grupo (`grabaciones/hola_bender/`, `grabaciones/comando_activar_secador/`, etc.).
- **Variar**: en cada serie, cambiar un poco el volumen, la velocidad y la distancia.

## 1. Frase de activacion: "Hola Bender"

Cada una de las 4 personas (Felipe, Laura, Maicol y Cristian):

| Que decir | Cuantas veces |
|---|---|
| "Hola Bender" | 30 (10 cerca, 10 a un metro, 10 a dos metros o con otro volumen) |

Ademas, cada persona dice **otras frases** que no son "Hola Bender" (son las negativas):

| Que decir | Cuantas veces |
|---|---|
| "Hola" (sola) | 5 |
| "Hola, como estas" | 5 |
| "Bender" (sola) | 5 |
| "Activar asistente" (frase parecida pero distinta, buen negativo) | 5 |
| Cualquier frase de 2 o 3 palabras del mismo tipo | 10 |

Total por persona: 30 de "Hola Bender" + 30 de otras frases = **60 tomas**.

## 2. Comandos para encender los actuadores

Primero se dice la frase de activacion, luego el comando. Hoy solo se entrenan y
reconocen por voz **bombillo** y **secador** (`motobomba` y `maquinilla` quedan
**pausados**: no se graba ni se entrena con ellos por ahora).

| Comando | Cuantas veces por persona |
|---|---|
| "Activar bombillo" | 5 |
| "Activar secador" | 5 |

## 3. Comandos para apagar los actuadores

| Comando | Cuantas veces por persona |
|---|---|
| "Apagar bombillo" | 5 |
| "Apagar secador" | 5 |

Total de comandos por persona: 4 comandos x 5 = **20 tomas**.

## 4. Respuestas de confirmacion (en tu propia voz)

Felipe y Laura graban una sola vez una frase corta de confirmacion (ej. "Listo"),
que el asistente reproduce tanto al reconocer "Hola Bender" como despues de
ejecutar un comando. Felipe aporta la voz de hombre (`confirmacion1.wav`) y
Laura la voz de mujer (`confirmacion2.wav`); ver README, seccion "Puesta en
marcha" paso 5.

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
| "Hola Bender" | 30 | 120 |
| Otras frases (negativas de persona) | 30 | 120 |
| Comandos activar/apagar (bombillo y secador) | 20 | 80 |
| Respuestas de confirmacion (solo Felipe y Laura) | 1 | 2 |
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

- Vocabulario completo de hora y fecha para Laura (le faltan 4 palabras:
  `hoy_estamos_a`, `de_la_manana`, `de_la_tarde`, `de_la_noche` — ya se le pidio
  grabarlas), y para Maicol y Cristian (todavia no tienen nada grabado).
- Grabaciones de persona/genero para identificar a quien no es Felipe o Laura
  (Maicol y Cristian) con el microfono de la placa, ademas de lo que ya cubre
  la seccion 1 y 2/3 (ver nota al inicio de este documento).
