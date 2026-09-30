# Asistente de voz local: ESP32-S3 + INMP441 + MAX98357A (MicroPython)

Detecta la frase de activación **"activar asistente"**, enciende un LED por 3 s
y responde reproduciendo una frase corta grabada en la propia voz del usuario.
Corre **100 % dentro del ESP32-S3**, sin internet, con el firmware oficial de
MicroPython (v1.29, sin recompilar nada, sin Arduino/ESP-IDF).

## Cómo funciona

1. **Captura**: el INMP441 entrega audio por I2S a 16 kHz. Se procesa en bloques de 32 ms.
2. **Análisis por bloque** (`audio_i2s.analizar`, ~5 ms): energía, cruces por cero y
   altas frecuencias, restando antes el offset de continua del micrófono
   (~-200, deriva entre bloques; sin restarlo el "silencio" parece ruido).
3. **VAD** (`vad.py`): detecta cuándo empieza y termina una frase por umbral de energía.
4. **Características** (`feature_extractor.py`): del segmento se recorta el silencio de los
   bordes y se arma un vector de 33 valores: amplitud en dB (16 puntos) + cruces por cero
   ponderados por amplitud (16 puntos) + duración.
5. **Modelo** (`inferencia.py`): ensamble de 5 CNN 1D pequeñas; se promedia su
   probabilidad (~50 ms por frase).
6. **Respuesta** (`salida_audio.py`): si la probabilidad supera `config.UMBRAL_CONFIANZA`,
   enciende el LED y reproduce `config.RUTA_CLIP_CONFIRMACION` (un `.wav` con la voz del
   usuario) por el amplificador MAX98357A. Mientras suena, el micrófono sigue acumulando
   audio (el eco de la propia respuesta); al terminar se descarta con
   `lector.vaciar_buffer()` antes de volver a escuchar, para no confundirlo con una orden.

El modelo se **entrena en la PC** (numpy puro, `entrenar_modelo.py`) y se copia a la
placa como un archivo `.py` con los pesos. La placa solo hace la inferencia — nunca
entrena ni necesita numpy/TensorFlow.

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

Pines evitados a propósito en todo el circuito: *strapping* (GPIO0, 3, 45, 46),
USB nativo (GPIO19, 20) y JTAG (GPIO39-42).

## Archivos

**En la placa** (MicroPython, sin librerías externas):
`config.py`, `audio_i2s.py`, `vad.py`, `feature_extractor.py`, `inferencia.py`,
`led.py`, `salida_audio.py`, `main.py`, `recolectar_datos.py`, `grabar_wav.py`,
`diagnostico_vad.py`, `modelo_pesos.py` (generado, no editar a mano), `confirmacion1.wav`
(o el nombre que tenga `config.RUTA_CLIP_CONFIRMACION`).

**En la PC** (no se copian a la placa; usan numpy/matplotlib):
`entrenar_modelo.py`, `generar_graficas.py`, `convertir_audio.py`.

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

### 3. Copiar los archivos de la placa

```
python -m mpremote connect COM3 cp config.py audio_i2s.py vad.py feature_extractor.py inferencia.py led.py salida_audio.py main.py recolectar_datos.py grabar_wav.py diagnostico_vad.py :
```

### 4. Grabar la frase de confirmación (con tu propia voz)

Dos formas:

- **Directo con el INMP441** (más simple, calidad más baja):
  ```python
  import grabar_wav
  grabar_wav.grabar("confirmacion1.wav", segundos=2)
  ```
- **Con otro dispositivo** (celular/PC, mejor calidad — recomendado): graba un WAV
  cualquiera y conviértelo al formato que necesita la placa (mono, PCM 16 bits;
  requiere `ffmpeg` instalado):
  ```
  python convertir_audio.py mi_grabacion.wav confirmacion1.wav
  ```
  Luego cópialo a la placa: `python -m mpremote connect COM3 cp confirmacion1.wav :`

  > `salida_audio.py` tolera encabezados WAV con bloques extra (`LIST`, etc.) que
  > agregan algunas apps de grabación — no asume que el audio empieza en el byte 44.

### 5. Grabar el dataset y entrenar el modelo

```python
import recolectar_datos as rd
rd.grabar(1, 20, "cerca")     # 1 = frase; distancias: cerca ~30 cm, media ~1 m, lejos ~2 m
rd.grabar(1, 15, "media")
rd.grabar(1, 15, "lejos")
rd.grabar(0, 20, "cerca")     # 0 = cualquier otra cosa (frases parecidas, ruidos, tos...)
rd.grabar(0, 15, "media")
rd.grabar(0, 15, "lejos")
rd.resumen(); rd.liberar()
```
Cada muestra se guarda al instante en `dataset.jsonl`. `rd.deshacer()` borra la última toma.

```
python -m mpremote connect COM3 cp :dataset.jsonl .
python entrenar_modelo.py            # ~2-4 min; --rapido para una grilla chica de prueba
python -m mpremote connect COM3 cp modelo_pesos.py :
```
`entrenar_modelo.py` compara MLP y CNN 1D (single y en ensamble) por validación
cruzada, elige el mejor y lo exporta. Imprime también una tabla de umbral de
decisión (detección vs. falsos positivos) para ajustar `config.UMBRAL_CONFIANZA`.

### 6. Ejecutar

```python
import main
main.ejecutar()
```

Consola esperada:
```
Listo. Esperando 'activar asistente'...
[DESCARTADO] sonido no coincide (confianza 0.12)
[COMANDO RECONOCIDO] 'activar asistente' (confianza 0.83).
Esperando 'activar asistente'...
```
(y tu voz respondiendo por el parlante en el momento del reconocimiento)

### 7. (Opcional) Gráficas de evaluación

```
python generar_graficas.py
```
Genera 6 imágenes en `graficas/`: matriz de confusión, curva ROC, efecto del
umbral, recall por distancia, comparación de arquitecturas y evolución v1→v2.

## Resultados (validación cruzada 5×3, 100 muestras: 50 frase + 50 otra)

| Versión | Precisión | Detecta la frase | Falsos positivos |
|---|---|---|---|
| v1: MLP, 60 muestras, solo envolvente lineal | ~73 % | - | - |
| v2 actual: ensamble de 5 CNN 1D | ~89 % (AUC 0.93) | 96 % (cerca 98 / 1 m 96 / 2 m 93) | ~17 % |

Se probaron ~15 variantes de características con 100 muestras, así que parte de la
mejora puede ser optimismo de selección. Lo que más ayudó: amplitud en escala
logarítmica, ponderar los cruces por cero por la amplitud y recortar menos silencio.
Las altas frecuencias (`hf`) no aportaron y no se usan, aunque se siguen guardando.

## Ajustes

- `config.UMBRAL_CONFIANZA` (por defecto **0.7**): más alto = menos falsos positivos y
  más frases perdidas (0.5 → 92 % detección / 16 % falsos positivos; 0.7 → 74 % / 10 %;
  0.9 → 52 % / 2 %). Ajustar según la tabla que imprime `entrenar_modelo.py`.
- Umbrales del VAD (`UMBRAL_RUIDO_INICIAL`, `FACTOR_ACTIVACION`): medidos con
  `diagnostico_vad.py`. Volver a medir si cambia el micrófono o el entorno.
- Si el amplificador no emite sonido con `bits=16`: es un problema conocido de
  MicroPython con el MAX98357A. `salida_audio.py` ya usa `bits=32` (muestra de 16
  bits desplazada a los bits altos), que sí funciona en placa.

## Límites y siguientes pasos

- Es un clasificador de una sola frase entrenado con la voz de un hablante y un
  entorno; no es reconocimiento de voz general. Más muestras (otras voces, otros
  ambientes, más frases parecidas como negativas) es la mejora con más impacto.
- El bucle principal de `main.py` no tiene un `try/except` general: si algo
  inesperado falla (lectura I2S, inferencia), el proceso se detiene y hay que
  reiniciar manualmente. Decisión deliberada por ahora, para que los fallos sean
  visibles durante las pruebas en hardware.
- Reconocer varias preguntas/comandos distintos (no solo "activar asistente") es
  el siguiente paso grande: requiere ampliar el dataset (una clase por pregunta,
  con bastantes más muestras) y adaptar el modelo a clasificación multi-clase.
