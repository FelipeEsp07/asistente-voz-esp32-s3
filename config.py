# Configuracion de hardware y de deteccion. Ajustar SOLO estos valores.

# --- Pines I2S hacia el INMP441 ---
# Se evitan pines de strapping (0, 3, 45, 46) y USB/JTAG (19, 20, 39-42).
PIN_SCK = 4   # BCLK del microfono
PIN_WS = 5    # WS / LRCLK del microfono
PIN_SD = 6    # Salida de datos del microfono (SD -> entrada del ESP32)

# --- LED indicador de activacion ---
PIN_LED = 7   # LED externo + resistencia (220-330 ohm) a GND

# --- Pines I2S hacia el amplificador MAX98357A (I2S(1, ...), separado del
# microfono que usa I2S(0, ...)) ---
PIN_AMP_BCLK = 15
PIN_AMP_LRC = 16
PIN_AMP_DIN = 17
RUTA_CLIP_CONFIRMACION = "confirmacion1.wav"  # frase corta grabada, en la propia voz

# --- Parametros de audio ---
SAMPLE_RATE = 16000
BLOCK_SAMPLES = 512          # ~32 ms por bloque a 16 kHz
I2S_BUFFER_BYTES = 20000     # buffer interno DMA del driver I2S

# --- Deteccion de actividad de voz (VAD) por energia ---
# La energia se mide SIN el offset de continua del INMP441 (audio_i2s.analizar
# resta la media de cada bloque). Medido con cableado directo: silencio con
# mediana ~7e12 y picos hasta ~5e13; voz cercana ~1e16; voz lejana ~1e14.
# Umbral = piso * factor = 5.6e13: por encima de los picos del silencio,
# muy por debajo de la voz.
UMBRAL_RUIDO_INICIAL = 7_000_000_000_000     # ~7e12: mediana de energia en silencio
FACTOR_ACTIVACION = 8.0               # energia > ruido_de_fondo * este factor => hay voz
BLOQUES_MIN_VOZ = 3                   # bloques consecutivos por encima del umbral para confirmar inicio
BLOQUES_MIN_SILENCIO = 14             # bloques de silencio para cerrar el segmento (~450ms,
                                       # para tolerar la pausa natural entre "activar" y "asistente")
MAX_BLOQUES_PALABRA = 90              # tope (~2.9 s), suficiente para la frase completa
MIN_BLOQUES_PALABRA = 4               # descarta ruidos/clics demasiado cortos

# --- Extraccion de caracteristicas ---
PUNTOS_ENVOLVENTE = 16                # puntos a los que se remuestrea cada canal del segmento
RECORTE_FRACCION_PICO = 0.05          # se descartan bloques iniciales/finales con amplitud
                                       # menor a esta fraccion del pico (cola de silencio);
                                       # 0.05 rindio mejor que 0.10/0.15 en validacion cruzada

# --- Modelo MLP ---
UMBRAL_CONFIANZA = 0.7                # probabilidad minima para aceptar "activar asistente".
                                       # Validacion cruzada: 0.5 -> detecta 92% de las frases,
                                       # 16% falsos positivos; 0.7 -> 74% / 10%; 0.9 -> 52% / 2%.

# --- LED ---
SEGUNDOS_LED_ENCENDIDO = 3            # tiempo que queda encendido tras detectar el comando

RUTA_DATASET = "dataset.jsonl"        # una muestra por linea, generado por recolectar_datos.py
