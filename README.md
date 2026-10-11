# Asistente de voz local: ESP32-S3 + INMP441 + MAX98357A (MicroPython)

Detecta la frase de activación **"activar asistente"**, enciende un LED por 3 s,
responde con una frase corta grabada en la propia voz del usuario y se queda
escuchando 10 segundos. En esa ventana reconoce:

- **"fecha y hora"**: dice la fecha y la hora actuales, armadas palabra por
  palabra con clips de voz pregrabados (sin texto-a-voz).
- **Comandos de actuadores**: "activar"/"apagar" para motobomba, bombillo,
  secador y maquinilla, que encienden o apagan un relé de 110 V.

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
5. **Modelo** (`inferencia.py`): ensamble de 5 CNN 1D con salida **softmax de 11 clases**
   (no binaria): `otro`, `activacion`, `fecha_hora`, y `activar_`/`apagar_` de
   `motobomba`, `bombillo`, `secador` y `maquinilla`. Se promedian las probabilidades
   de los 5 miembros del ensamble.
6. **Estados** (`asistente.py`):
   - **ESPERA**: solo acepta la clase `activacion` (probabilidad ≥ `config.UMBRAL_CONFIANZA`).
     Cualquier otra cosa se descarta, aunque se "parezca" a una pregunta o comando — las
     preguntas y comandos **no se evalúan** si el asistente no está activo.
   - Al activarse: LED encendido 3 s, suena la confirmación (`confirmacion1.wav`,
     la voz del usuario) y entra en **ESCUCHA** durante `config.SEGUNDOS_ESCUCHA` (10 s).
   - **ESCUCHA**: evalúa las 11 clases. Si la más probable es `fecha_hora` y supera
     `config.UMBRAL_PREGUNTA`, responde la fecha y la hora (`hora.py`). Si es un comando
     de actuador, lo ejecuta (`actuadores.py` + `comandos.py`) y confirma con el mismo
     clip de voz. Si no es ninguna de las dos, **no cierra la escucha**: sigue esperando
     hasta que se acaben los 10 s (`[ESCUCHA] sin pregunta reconocida: ...` en consola,
     con la probabilidad de cada clase, útil para depurar).
   - Tras cada sonido propio (confirmación o respuesta), hay una ventana de 0.8 s sin
     escuchar (`MS_SORDO_TRAS_RESPUESTA`) para que el eco por el parlante no se
     reconozca como una orden nueva.
7. **Fecha y hora** (`hora.py`): no hay texto-a-voz. La frase se arma concatenando
   clips de `clips/<persona>_<palabra>.wav` (por ejemplo "hoy estamos a" + "cuatro" +
   "de" + "octubre" + "y" + "son las" + "siete" + "y" + "media" + "de la tarde").
   La placa **no tiene reloj con batería**: `machine.RTC()` vuelve a arrancar en el
   año 2000 cada vez que se desconecta la energía. Si no se fijó la hora, el
   asistente avisa en consola y no responde la pregunta. Ver `fijar_hora.py`.
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
| Módulo de relés | IN1 → motobomba | **GPIO11** | |
| Módulo de relés | IN2 → bombillo | **GPIO12** | |
| Módulo de relés | IN3 → secador | **GPIO13** | pin provisional, confirmar cableado |
| Módulo de relés | IN4 → maquinilla | **GPIO14** | pin provisional, confirmar cableado |

**Lado de 110V de cada canal** (motobomba, bombillo, secador, maquinilla): usar los
bornes **COM** y **NO** (normalmente abierto, así el equipo queda apagado sin
energía). Fase → COM, NO → un terminal del equipo, el otro terminal del equipo → neutro.

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
a mano), `confirmacion1.wav` (frase de confirmación, en la voz del usuario),
`clips/` (vocabulario para armar la respuesta de fecha y hora), `grabar_placa.py`
(graba tomas con el micrófono de la placa, para ampliar el dataset).

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
`tono.py`, `modelo_pesos_binario_anterior.py`, `dataset_inmp441_viejo.jsonl`.
`confirmacion2.wav` es una toma suelta sin usar (`config.RUTA_CLIP_CONFIRMACION`
apunta a `confirmacion1.wav`).

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
python -m mpremote connect COM3 cp config.py audio_i2s.py vad.py feature_extractor.py inferencia.py led.py salida_audio.py actuadores.py comandos.py hora.py asistente.py main.py grabar_placa.py modelo_pesos.py confirmacion1.wav :
python -m mpremote connect COM3 cp -r clips :
```

### 4. Fijar la hora

La placa no tiene reloj con batería: cada vez que se desconecta vuelve al año
2000. Antes de probar "fecha y hora", en cada sesión:

```
python fijar_hora.py COM3
```

### 5. Grabar la frase de confirmación (con tu propia voz)

```
python grabar_wav.py felipe confirmacion 3
copy grabaciones\confirmacion\felipe_confirmacion_01.wav confirmacion1.wav
python -m mpremote connect COM3 cp confirmacion1.wav :
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

El modelo reconoce 11 clases: `otro`, `activacion`, `fecha_hora`, y
`activar_`/`apagar_` de `motobomba`, `bombillo`, `secador` y `maquinilla`.
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
  grabar_placa.tomas("activar_motobomba", 30)
  ```
  y luego, desde la PC, `python copiar_grabaciones_placa.py COM3 activar_motobomba <persona> audios/comandos`
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
Listo. Esperando 'activar asistente'...
[DESCARTADO] sonido no coincide (activacion 0.12)
[COMANDO RECONOCIDO] 'activar asistente' (confianza 0.93).
Escuchando 10 segundos...
[PREGUNTA] fecha_hora (confianza 0.81)
Esperando 'activar asistente'...
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

Último entrenamiento exportado a `modelo_pesos.py` (381 muestras, validación
cruzada 5×3, ensamble de 5 CNN 1D de 8 filtros):

| Clase | Recall |
|---|---|
| otro | 0.75 |
| activación ("activar asistente") | 0.93 (AUC 0.98) |
| fecha y hora | 0.79 |
| activar motobomba | 0.85 |
| activar bombillo | 0.92 |

Precisión multiclase: 0.819. Con `config.UMBRAL_CONFIANZA = 0.5`, la activación
se detecta el 84-87 % de las veces con 3-5 % de falsos positivos.

**Este resultado ya está desactualizado**: desde entonces se grabaron `apagar_bombillo`,
`apagar_motobomba` y `activar_secador`/`apagar_secador` (ver `GUIA_GRABACIONES.md`
y el estado de `audios/`), pero el modelo todavía no se reentrenó con esas clases.
Hay que repetir el paso 8 antes de usar esos comandos por voz.

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
  en los cuatro). Bajarlo si el equipo no debe quedar tanto tiempo encendido.

## Límites y siguientes pasos

- **Reentrenar con las clases nuevas**: `apagar_bombillo`, `apagar_motobomba`,
  `activar_secador` y `apagar_secador` ya tienen audio grabado pero el modelo en
  la placa no las conoce todavía (ver "Resultados"). `activar_maquinilla` y
  `apagar_maquinilla` todavía no tienen ninguna grabación.
- **Pines del secador y la maquinilla sin confirmar**: GPIO13 y GPIO14 son
  provisionales, a la espera de cablear esos relés.
- **Varias voces, pero solo para los comandos**: el vocabulario de fecha y hora
  (`clips/`) hoy solo tiene la voz de Felipe (`clips/felipe_*.wav`); el nombre de
  archivo ya deja espacio para otras personas (`clips/<persona>_<palabra>.wav`),
  pero falta grabarlas y que `asistente.py` elija la voz según quién habló.
- **No hay identificación de quién habla**: cualquier persona activa el
  asistente y ejecuta comandos; no se sabe si es Felipe, Laura, Maicol o
  Cristian. Es el siguiente paso grande si se quiere personalizar la respuesta.
- `GUIA_GRABACIONES.md` describe frases y cantidades de una etapa anterior del
  proyecto (la frase de activación "Hola PAN", preguntas distintas a las
  actuales); revisarla antes de seguirla al pie de la letra.
- El bucle principal de `asistente.py` no tiene un `try/except` general: si algo
  inesperado falla (lectura I2S, inferencia), el proceso se detiene y hay que
  reiniciar manualmente. Decisión deliberada por ahora, para que los fallos sean
  visibles durante las pruebas en hardware.
