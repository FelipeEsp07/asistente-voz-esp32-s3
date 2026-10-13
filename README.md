# Asistente de voz local: ESP32-S3 + INMP441 + MAX98357A (MicroPython)

Detecta la frase de activación **"Hola Bender"**, enciende un LED por 3 s,
responde con una frase corta grabada en la voz de quien habló (o una voz
acorde a su género si no se le conoce) y se queda escuchando 10 segundos.
En esa ventana reconoce:

- **"fecha y hora"**: dice la fecha y la hora actuales, armadas palabra por
  palabra con clips de voz pregrabados (sin texto-a-voz), con la voz de quien
  preguntó.
- **Comandos de actuadores**: "activar"/"apagar" para **bombillo** y
  **secador**, que encienden o apagan un relé de 110 V. `motobomba` y
  `maquinilla` quedan pausados por ahora (ver "Cómo funciona").

Corre **100 % dentro del ESP32-S3**, sin internet, con el firmware oficial de
MicroPython (v1.29, sin recompilar nada, sin Arduino/ESP-IDF). El entrenamiento
del modelo se hace en la PC; la placa solo hace la inferencia.

## Cómo funciona

1. **Captura**: el INMP441 entrega audio por I2S a 16 kHz. Se procesa en bloques de 32 ms.
2. **Análisis por bloque** (`audio_i2s.analizar`, ~5 ms): energía, cruces por cero y
   altas frecuencias, restando antes el offset de continua del micrófono
   (~-200, deriva entre bloques; sin restarlo el "silencio" parece ruido).
3. **VAD** (`vad.py`): detecta cuándo empieza y termina una frase por umbral de energía.
4. **Características** (`feature_extractor.py`): del segmento se recorta el silencio de los
   bordes y se arma un vector de 33 valores: amplitud en dB (16 puntos) + cruces por cero
   ponderados por amplitud (16 puntos) + duración. Es el mismo vector en la placa y en la PC.
5. **Modelo** (`inferencia.py`, `modelo_pesos.py` con `TIPO = "cnn_cascada"`): no es
   un clasificador plano de una sola salida, sino una **cascada de 5 grupos**, cada
   uno un ensamble de CNN 1D con softmax, evaluados sobre el mismo vector de
   características (entrenados por `entrenar_modelo.py`):
   - **intención**: `otro` / `activacion` / `fecha_hora` / `comando`. Decide si hay
     que responder algo y de qué tipo.
   - **equipo**: a qué actuador se refiere el comando. Por ahora solo `bombillo` y
     `secador` — `motobomba` y `maquinilla` quedan **pausados**: tienen pin reservado
     en `config.ACTUADORES` pero no se entrenan ni se reconocen por voz, para
     enfocar los datos y el entrenamiento en dos actuadores primero.
   - **verbo**: `activar` / `apagar`.
   - **persona**: quién habló, entre las 4 personas con grabaciones propias —
     `felipe`, `laura`, `maicol`, `cristian`.
   - **género**: `hombre` / `mujer`, para cuando quien habla no es ninguna de esas
     4 personas (así igual se puede elegir una voz de confirmación acorde).
   Los grupos "equipo" y "verbo" solo se consultan cuando "intención" dijo
   `comando`; "persona" y "género" se consultan siempre, para elegir la voz de
   la respuesta. Se promedian las probabilidades de los miembros de cada ensamble.
6. **Estados** (`asistente.py`):
   - **ESPERA**: solo acepta la clase `activacion` del grupo "intención"
     (probabilidad ≥ `config.UMBRAL_CONFIANZA`). Cualquier otra cosa se descarta,
     aunque se "parezca" a una pregunta o comando — las preguntas y comandos
     **no se evalúan** si el asistente no está activo.
   - Al activarse: LED encendido 3 s, suena la confirmación y entra en **ESCUCHA**
     durante `config.SEGUNDOS_ESCUCHA` (10 s).
   - **Voz de confirmación**: se identifica a quién habló (grupo "persona"); si es
     una de las 4 personas conocidas se usa su género (`felipe`/`maicol`/`cristian`
     → hombre, `laura` → mujer); si no, se usa el grupo "género" sobre esa misma
     grabación. Con género "hombre" se reproduce `confirmacion1.wav` (voz de
     Felipe); con "mujer", `confirmacion2.wav` (voz de Laura).
   - **ESCUCHA**: evalúa de nuevo "intención". Si la más probable es `fecha_hora` y
     supera `config.UMBRAL_PREGUNTA`, responde la fecha y la hora (`hora.py`), con
     la voz de quien preguntó. Si es `comando`, consulta "equipo" y "verbo", lo
     ejecuta (`actuadores.py` + `comandos.py`) y confirma con la misma regla de voz
     de arriba. Si no es ninguna de las dos, **no cierra la escucha**: sigue
     esperando hasta que se acaben los 10 s (`[ESCUCHA] sin pregunta reconocida: ...`
     en consola, con la probabilidad de cada clase, útil para depurar).
   - Tras cada sonido propio (confirmación o respuesta), hay una ventana de 0.8 s sin
     escuchar (`MS_SORDO_TRAS_RESPUESTA`) para que el eco por el parlante no se
     reconozca como una orden nueva.
7. **Fecha y hora** (`hora.py`): no hay texto-a-voz. La frase se arma concatenando
   clips de `clips/<persona>_<palabra>.wav` (por ejemplo "hoy estamos a" + "cuatro" +
   "de" + "octubre" + "y" + "son las" + "siete" + "y" + "media" + "de la tarde"),
   con la voz de la persona identificada. Hoy solo **Felipe** tiene el vocabulario
   completo (68 palabras); **Laura** le faltan 4 (`hoy_estamos_a`, `de_la_manana`,
   `de_la_tarde`, `de_la_noche`). Maicol y Cristian todavía no tienen vocabulario
   grabado.
   La placa **no tiene reloj con batería**: `machine.RTC()` vuelve a arrancar en el
   año 2000 cada vez que se desconecta la energía. Si no se fijó la hora, el
   asistente avisa en consola y no responde la pregunta. Hay que fijarla con
   `fijar_hora.py` en cada sesión.
8. **Actuadores** (`actuadores.py`): cada uno tiene un GPIO y un tiempo máximo
   de encendido; se apaga solo aunque no se diga "apagar" o el programa falle.
   Los pines se crean ya apagados, para que ningún relé se active al encender la placa.

## Cableado completo

**NO usar la breakout Freenove para el micrófono** — se probó y degrada mucho la
señal del INMP441 (ver notas de calibración en `diagnostico_vad.py`). Cablear todo
directo, con jumpers cortos (menos de 20 cm en las líneas I2S).

El micrófono usa el primer periférico I2S (`I2S(0, ...)`) y el amplificador el
segundo (`I2S(1, ...)`) — son independientes, no chocan entre sí.

| Componente | Pin | ESP32-S3 | Nota |
|---|---|---|---|
| INMP441 | VDD | 3V3 | |
| INMP441 | GND | GND | |
| INMP441 | L/R | GND | fija canal izquierdo (mono) |
| INMP441 | SCK (BCLK) | **GPIO4** | |
| INMP441 | WS (LRCLK) | **GPIO5** | |
| INMP441 | SD (DOUT) | **GPIO6** | |
| LED | ánodo (vía resistencia 220-330 Ω) | **GPIO7** | |
| LED | cátodo | GND | |
| MAX98357A | VIN | 5V (USB) | con 3V3 enciende pero suena más flojo |
| MAX98357A | GND | GND | |
| MAX98357A | BCLK | **GPIO15** | |
| MAX98357A | LRC (WS) | **GPIO16** | |
| MAX98357A | DIN | **GPIO17** | |
| MAX98357A | SD | sin conectar | flotando = habilitado, mezcla (L+R)/2 |
| MAX98357A | GAIN | sin conectar | 9 dB por defecto |
| Parlante (4-8 Ω, 3 W) | — | SPK+ / SPK− del MAX98357A | salida en puente: **ningún cable a GND** |
| Módulo de relés (4 canales, 5V, optoacoplador) | VCC | **5V (USB)** | con 3V3 no conmuta: la bobina necesita 5V reales |
| Módulo de relés | GND | GND | tierra compartida con la placa, obligatoria |
| Módulo de relés | IN1 → motobomba | **GPIO11** | pin reservado, **pausado**: no se entrena ni se reconoce por voz por ahora |
| Módulo de relés | IN2 → bombillo | **GPIO12** | activo |
| Módulo de relés | IN3 → secador | **GPIO13** | activo; pin provisional, confirmar cableado |
| Módulo de relés | IN4 → maquinilla | **GPIO14** | pin reservado, **pausado**: no se entrena ni se reconoce por voz por ahora |

**Lado de 110V de cada canal** (motobomba, bombillo, secador, maquinilla): usar los
bornes **COM** y **NO** (normalmente abierto, así el equipo queda apagado sin
energía). Fase → COM, NO → un terminal del equipo, el otro terminal del equipo → neutro.
Motobomba y maquinilla quedan pausados para enfocar los datos y el entrenamiento
en bombillo y secador primero; sus pines y cableado quedan listos para cuando se
reactiven.

**Lógica del módulo (comprobada en placa): nivel BAJO activa el relé.** `actuadores.py`
ya usa `NIVEL_ENCENDIDO = 0` / `NIVEL_APAGADO = 1`, y los pines se crean ya apagados
para que ningún relé se active al encender la placa.

Pines evitados a propósito en todo el circuito: *strapping* (GPIO0, 3, 45, 46),
USB nativo (GPIO19, 20) y JTAG (GPIO39-42).

## Archivos

**En la placa** (MicroPython, sin librerías externas):
`config.py`, `audio_i2s.py`, `vad.py`, `feature_extractor.py`, `inferencia.py`,
`led.py`, `salida_audio.py`, `actuadores.py`, `comandos.py`, `hora.py`,
`asistente.py` (lógica del asistente), `main.py` (punto de entrada: solo importa
`ejecutar`, no arranca nada al encender), `modelo_pesos.py` (generado, no editar
a mano), `confirmacion1.wav` / `confirmacion2.wav` (frase de confirmación en voz
de hombre/mujer — Felipe y Laura respectivamente —, elegida según quién habló o
su género), `clips/` (vocabulario para armar la respuesta de fecha y hora, por
persona), `grabar_placa.py` (graba tomas con el micrófono de la placa, para
ampliar el dataset).

**En la PC** (no se copian a la placa; usan numpy, matplotlib, sounddevice y soundfile):
`entrenar_modelo.py` (entrena y exporta el modelo), `preparar_dataset_pc.py`
(convierte WAV a `dataset.jsonl`), `preparar_clips.py` (recorta el vocabulario a
clips), `grabar_wav.py` (tomas de entrenamiento o vocabulario con el micrófono
de la PC), `grabar_lote.py` (grabación rápida por lotes, con aviso sonoro),
`copiar_grabaciones_placa.py` (pasa a la PC lo grabado con `grabar_placa.py`),
`fijar_hora.py` (fija el reloj de la placa con la hora de la PC), `generar_graficas.py`,
`convertir_audio.py`.

**Heredados de la versión binaria anterior** (ya no forman parte del flujo
actual, se conservan como referencia): `recolectar_datos.py`, `diagnostico_vad.py`,
`modelo_pesos_binario_anterior.py`.

(`tono.py` **no** es heredado, aunque antes estaba en esta lista: lo usan
activamente `preparar_dataset_pc.py`, `entrenar_modelo.py` y `asistente.py`
para el tono de quien habla, grupos "persona"/"género" — ver "Cómo funciona".)

## Puesta en marcha

### 1. Flashear MicroPython

Firmware oficial ESP32-S3 con soporte Octal-SPIRAM (v1.20+, necesaria para
`machine.I2S`): https://micropython.org/download/ESP32_GENERIC_S3/

```
python -m esptool --port COM3 erase-flash
python -m esptool --port COM3 write-flash 0x0 ESP32_GENERIC_S3-SPIRAM_OCT-<version>.bin
```

### 2. Cablear el circuito

Según la tabla de arriba. Verifica la placa detectada antes de continuar:
`python -m esptool --port COM3 chip-id` (confirma el MAC si has usado más de una placa).

### 3. Copiar los archivos a la placa

```
python -m mpremote connect COM3 cp config.py audio_i2s.py vad.py feature_extractor.py inferencia.py led.py salida_audio.py actuadores.py comandos.py hora.py asistente.py main.py grabar_placa.py modelo_pesos.py confirmacion1.wav confirmacion2.wav :
python -m mpremote connect COM3 cp -r clips :
```

### 4. Fijar la hora

La placa no tiene reloj con batería: cada vez que se desconecta vuelve al año
2000. Antes de probar "fecha y hora", en cada sesión:

```
python fijar_hora.py COM3
```

### 5. Grabar la frase de confirmación (con tu propia voz)

Una toma por voz: `confirmacion1.wav` (voz de hombre, hoy la de Felipe) y
`confirmacion2.wav` (voz de mujer, hoy la de Laura).

```
python grabar_wav.py felipe confirmacion 3
copy grabaciones\confirmacion\felipe_confirmacion_01.wav confirmacion1.wav
python grabar_wav.py laura confirmacion 3
copy grabaciones\confirmacion\laura_confirmacion_01.wav confirmacion2.wav
python -m mpremote connect COM3 cp confirmacion1.wav confirmacion2.wav :
```

> `salida_audio.py` tolera encabezados WAV con bloques extra (`LIST`, etc.) que
> agregan algunas apps de grabación — no asume que el audio empieza en el byte 44.

### 6. Grabar el vocabulario de fecha y hora

Son palabras sueltas (números, meses, días, "son las", "y", "de la tarde"...)
que `hora.py` concatena para responder. Se grabó con el micrófono de la PC y se
recorta solo a la voz, para que la respuesta no tenga silencios largos:

```
python grabar_wav.py felipe --vocabulario
python preparar_clips.py felipe
python -m mpremote connect COM3 cp -r clips :
```

### 7. Grabar el dataset de entrenamiento

El modelo es una cascada de 5 grupos (ver "Cómo funciona"): `intención` (`otro`,
`activacion`, `fecha_hora`, `comando`), `equipo` (hoy solo `bombillo` y
`secador`), `verbo` (`activar`/`apagar`), `persona` (`felipe`, `laura`, `maicol`,
`cristian`) y `género` (`hombre`/`mujer`). Las grabaciones de comando se
etiquetan como `activar_bombillo`, `apagar_bombillo`, `activar_secador`,
`apagar_secador`; no hace falta grabar nada aparte para "persona" o "género" —
`preparar_dataset_pc.py` ya etiqueta cada toma (de activación, comandos o
preguntas de hora/fecha) con el nombre de quien habló, y de ahí sale también
su género (`entrenar_modelo.py`).

Dos formas de grabar, y conviene usar la segunda para todo lo que la placa
tiene que reconocer en uso real:

- **Con el micrófono de la PC** (`grabar_wav.py`, `grabar_lote.py`): más cómodo,
  pero el micrófono de la PC suena distinto al INMP441 de la placa. En las
  pruebas, un modelo entrenado solo con audio de la PC detectaba mal la misma
  frase dicha ante la placa.
- **Con el micrófono de la placa** (recomendado): en el REPL, sin el asistente
  corriendo,
  ```python
  import grabar_placa
  grabar_placa.tomas("activar_bombillo", 30)
  ```
  y luego, desde la PC, `python copiar_grabaciones_placa.py COM3 activar_bombillo <persona> audios/comandos`
  (la placa tiene poco espacio: borra las tomas ya copiadas antes de seguir grabando).

Para que el modelo no aprenda "a qué micrófono" en vez de "qué palabra", conviene
grabar cada clase con **varias personas** (en este proyecto: Felipe, Laura,
Maicol y Cristian) y mantener el número de tomas parecido entre clases.

### 8. Preparar el dataset y entrenar

```
python preparar_dataset_pc.py audios/fecha_hora --preguntas --hablante <persona>
python preparar_dataset_pc.py audios/otro --label 0
python preparar_dataset_pc.py audios/comandos --preguntas --hablante <persona>
python entrenar_modelo.py            # unos minutos; --rapido para una grilla chica de prueba
python -m mpremote connect COM3 cp modelo_pesos.py :
```
`preparar_dataset_pc.py` reconstruye `dataset.jsonl` con el mismo VAD y las
mismas características que usa la placa, para que el entrenamiento sea fiel a
lo que se va a ejecutar. `entrenar_modelo.py` compara una grilla de CNN 1D y su
ensamble por validación cruzada, elige el mejor y lo exporta.

### 9. Ejecutar

La placa no arranca el asistente al encenderse: queda en el REPL. Para iniciarlo:

```python
import main
main.ejecutar()
```

Consola esperada:
```
Listo. Esperando 'Hola Bender'...
[DESCARTADO] sonido no coincide (activacion 0.12)
[COMANDO RECONOCIDO] 'Hola Bender' (confianza 0.93).
Escuchando 10 segundos...
[PREGUNTA] fecha_hora (confianza 0.81)
Esperando 'Hola Bender'...
```
(y la respuesta sonando por el parlante en el momento del reconocimiento)

### 10. (Opcional) Gráficas de evaluación

```
python generar_graficas.py
```
Genera 6 imágenes en `graficas/`: matriz de confusión, curva ROC, efecto del
umbral, recall por distancia, comparación de arquitecturas y evolución v1→v2.
Corresponden a la versión binaria anterior; falta una versión para el modelo
multiclase actual.

## Resultados

**Desactualizados**: el último número publicado aquí (381 muestras, precisión
multiclase 0.819) corresponde a una versión anterior a la cascada de 5 grupos y
a cuando `motobomba` y `maquinilla` todavía se entrenaban. Con la arquitectura
actual (`intención` / `equipo` / `verbo` / `persona` / `género`) y el equipo
activo reducido a `bombillo` y `secador`, hay que repetir el paso 8 y tomar la
tabla que imprime `entrenar_modelo.py` para cada grupo (`acc=...` por grupo al
final del entrenamiento) como referencia vigente.

## Ajustes

- `config.UMBRAL_CONFIANZA` (activación) y `config.UMBRAL_PREGUNTA` (preguntas y
  comandos), ambos en **0.5** por defecto: más alto = menos falsos positivos y
  más frases perdidas. Ajustar según la tabla que imprime `entrenar_modelo.py`.
- `config.SEGUNDOS_ESCUCHA` (10 s): tiempo de la ventana después de la activación.
- Umbrales del VAD (`UMBRAL_RUIDO_INICIAL`, `FACTOR_ACTIVACION`): medidos con
  `diagnostico_vad.py`. Volver a medir si cambia el micrófono o el entorno.
- Si el amplificador no emite sonido con `bits=16`: es un problema conocido de
  MicroPython con el MAX98357A. `salida_audio.py` ya usa `bits=32` (muestra de 16
  bits desplazada a los bits altos), que sí funciona en placa.
- `config.ACTUADORES`: tiempo máximo de encendido por equipo (900 s, provisional
  en los cuatro, aunque solo `bombillo` y `secador` se reconocen por voz hoy).
  Bajarlo si el equipo no debe quedar tanto tiempo encendido.

## Límites y siguientes pasos

- **`motobomba` y `maquinilla` pausados**: tienen pin reservado en
  `config.ACTUADORES` pero el grupo "equipo" ya no se entrena con sus clases
  (`EQUIPO_CLASES` en `entrenar_modelo.py` solo tiene `bombillo` y `secador`).
  Para reactivarlos hay que volver a grabar y sumarlos a esa lista.
- **Pines del secador sin confirmar**: GPIO13 es provisional, a la espera de
  cablear ese relé.
- **Vocabulario de fecha y hora incompleto**: `clips/` solo tiene el set
  completo de Felipe (68 palabras); a Laura le faltan 4 (`hoy_estamos_a`,
  `de_la_manana`, `de_la_tarde`, `de_la_noche`) y a Maicol y Cristian les falta
  todo el vocabulario. Sin esas grabaciones, `hora.py` no puede responder con
  su voz.
- **Identificación de persona y género en desarrollo**: la cascada ya define
  los grupos "persona" y "género" y `modelo_pesos.py` se regenera con ellos al
  reentrenar, pero falta que `asistente.py`/`audio_i2s.py` queden terminados de
  integrar la elección de `confirmacion1.wav`/`confirmacion2.wav` y de la voz
  de la respuesta de fecha y hora según quién habló.
- El bucle principal de `asistente.py` no tiene un `try/except` general: si algo
  inesperado falla (lectura I2S, inferencia), el proceso se detiene y hay que
  reiniciar manualmente. Decisión deliberada por ahora, para que los fallos sean
  visibles durante las pruebas en hardware.
