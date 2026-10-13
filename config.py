# Configuracion de hardware y de deteccion. Ajustar SOLO estos valores.

# --- Frase de activacion del asistente ---
PALABRA_ACTIVACION = "Hola Bender"

# --- Pines I2S hacia el INMP441 ---
# Se evitan pines de strapping (0, 3, 45, 46) y USB/JTAG (19, 20, 39-42).
PIN_SCK = 4   # BCLK del microfono
PIN_WS = 5    # WS / LRCLK del microfono
PIN_SD = 6    # Salida de datos del microfono (SD -> entrada del ESP32)

# --- LED indicador de activacion ---
PIN_LED = 7   # LED externo + resistencia (220-330 ohm) a GND

# --- Actuadores (reles de 110 V; el ESP32 solo controla la señal de 3.3 V) ---
# {nombre: (GPIO, segundos_maximos_encendido)}. El tiempo maximo apaga el equipo
# aunque el programa falle. Valores provisionales: ajustar por equipo.
# NOTA: por ahora solo "bombillo" y "secador" estan activos (se entrenan y se
# reconocen por voz). "motobomba" y "maquinilla" quedan pausados: el pin queda
# reservado aqui, pero el grupo "actuador" del modelo (ver entrenar_modelo.py,
# ACTUADOR_CLASES) ya no se entrena con ellos, para enfocar los datos y el
# entrenamiento en dos actuadores primero.
ACTUADORES = {
    "motobomba": (11, 900),   # IN1 del modulo de reles (pausado, ver nota arriba)
    "bombillo": (12, 900),    # IN2 del modulo de reles
    "secador": (13, 900),     # IN3 del modulo de reles (pin provisional, confirmar cableado)
    "maquinilla": (14, 900),  # IN4 del modulo de reles (pausado, ver nota arriba; pin provisional, confirmar cableado)
}
SEGUNDOS_VIGILANCIA = 5  # watchdog de hardware: reinicia la placa si el programa se congela

# --- Comandos de voz: {etiqueta: (actuador, "on" o "off")} ---
# Se dice la frase de activacion, luego la palabra "comandos" y luego el
# comando. Las etiquetas son directamente las clases del grupo "actuador"
# del modelo en cascada (inferencia.py). Los de motobomba/maquinilla quedan
# aqui para cuando se reactiven esos actuadores, pero hoy el grupo "actuador"
# no los reconoce (ver nota junto a ACTUADORES).
COMANDOS = {
    "activar_motobomba": ("motobomba", "on"),
    "activar_bombillo": ("bombillo", "on"),
    "apagar_bombillo": ("bombillo", "off"),
    "apagar_motobomba": ("motobomba", "off"),
    "activar_secador": ("secador", "on"),
    "apagar_secador": ("secador", "off"),
    "activar_maquinilla": ("maquinilla", "on"),
    "apagar_maquinilla": ("maquinilla", "off"),
}
SEGUNDOS_ESCUCHA = 10  # tiempo para decir la pregunta despues de la frase de activacion

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
UMBRAL_CONFIANZA = 0.5                # probabilidad minima para aceptar PALABRA_ACTIVACION.
UMBRAL_PREGUNTA = 0.5                 # probabilidad minima, dentro de la ventana ESCUCHA, para aceptar
                                       # "fecha_hora" o la palabra "comandos" (grupo "intencion").
                                       # Validacion cruzada: 0.5 -> detecta 92% de las frases,
                                       # 16% falsos positivos; 0.7 -> 74% / 10%; 0.9 -> 52% / 2%.
UMBRAL_ACTUADOR = 0.5                 # probabilidad minima, dentro de ESCUCHA_ACTUADOR, para ejecutar
                                       # uno de los 4 comandos del grupo "actuador". No hay clase "otro"
                                       # en ese grupo: un umbral muy bajo puede activar un actuador con
                                       # un sonido que no era ninguno de los 4 comandos.

# --- LED ---
SEGUNDOS_LED_ENCENDIDO = 3            # tiempo que queda encendido tras detectar el comando

MS_PAUSA_ENTRE_CLIPS = 80             # pausa entre palabras al decir la fecha/hora; sin esto
                                       # las palabras bien recortadas (sin margen de silencio)
                                       # se pegan y suenan cortadas

RUTA_DATASET = "dataset.jsonl"        # una muestra por linea, generado por recolectar_datos.py

# --- Identificacion de quien habla (persona/genero), al detectar la activacion ---
UMBRAL_PERSONA = 0.5                  # confianza minima para considerar reconocida a una
                                       # de las 4 personas conocidas; si no se alcanza, se
                                       # elige el clip de confirmacion solo por genero.
