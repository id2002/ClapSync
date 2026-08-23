clap_sync.py — Detecta automáticamente el desfase de audio/video usando un aplauso (clap)
como referencia, y te dice cuánto y en qué dirección mover el audio en OBS o DaVinci Resolve.
 **** OJO - Todvia esta en desarrollo ****
CÓMO FUNCIONA
-------------
1. Extrae el audio del video con ffmpeg.
2. Busca el "golpe" del aplauso en el audio: el instante donde la energía sube de golpe
   (transiente), no solo el punto más fuerte (para evitar confundirlo con otro ruido fuerte).
3. Busca el "golpe" del aplauso en el video: el instante donde hay más cambio de un frame
   a otro (las manos juntándose es el movimiento más brusco de la toma).
4. Compara ambos instantes y te da el offset en milisegundos y en frames.

REQUISITOS (instalar una sola vez)
-----------------------------------
    pip install numpy opencv-python scipy --break-system-packages
    (además necesitas ffmpeg instalado y accesible desde la terminal: prueba `ffmpeg -version`)

USO
---
    python clap_sync.py mi_video.mp4

    # Si el aplauso no es el sonido más fuerte del clip, o hay ruido de fondo,
    # acota la búsqueda a la ventana donde SABES que ocurre el aplauso (en segundos):
    python clap_sync.py mi_video.mp4 --start 0 --end 3

LIMITACIONES (para que no te lleves sorpresas)
-----------------------------------------------
- Necesita que el aplauso sea un sonido corto y notablemente más fuerte que lo que lo rodea.
- Necesita que el movimiento de aplaudir sea el movimiento más brusco dentro de la ventana
  analizada (evita mover otras cosas frente a cámara justo antes/después del clap).
- Es una estimación heurística: siempre revisa el resultado escuchando/mirando ese punto
  antes de aplicarlo a un proyecto largo.
"""
