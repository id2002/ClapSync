clap_sync_gui.py — Detecta automáticamente el desfase de audio/video usando un aplauso (clap)
como referencia, y te dice cuánto y en qué dirección mover el audio en OBS o DaVinci Resolve.
 **** OJO - Todvia esta en desarrollo ****
CÓMO FUNCIONA
-------------
1. Extrae el audio del video con ffmpeg.
2. Busca el "golpe" del aplauso en el audio: atenúa voz y graves, y detecta el ataque corto
   de alta frecuencia. El umbral usa mediana/MAD, por lo que música o un golpe aislado no
   elevan artificialmente el umbral de todo el clip.
3. Busca el "golpe" del aplauso en el video: el instante donde hay más cambio de un frame
   a otro (las manos juntándose es el movimiento más brusco de la toma).
4. Conserva las parejas que comparten el mismo desfase temporal; así un movimiento ajeno al
   aplauso no debería terminar emparejado por casualidad.

REQUISITOS (instalar una sola vez)
-----------------------------------
    pip install numpy opencv-python scipy --break-system-packages
    (además necesitas ffmpeg instalado y accesible desde la terminal: prueba `ffmpeg -version`)

USO
---
    python clap_sync_gui.py mi_video.mp4

    # Si el aplauso no es el sonido más fuerte del clip, o hay ruido de fondo,
    # acota la búsqueda a la ventana donde SABES que ocurre el aplauso (en segundos):
    python clap_sync_gui.py mi_video.mp4 --start 0 --end 3

    # Si las manos ocupan una zona pequeña, analiza solo esa zona. Las coordenadas
    # son fracciones del cuadro: x0,y0,x1,y1.
    python clap_sync_gui.py mi_video.mp4 --start 0 --end 3 --crop 0.30,0.25,0.70,0.85

En la ventana gráfica puedes escribir el mismo recorte en "Recorte manual". Es la opción
más útil si hay personas, cámara o fondo moviéndose: encierra solo las manos y usa una
ventana corta de 2–4 segundos alrededor del aplauso.

Si el audio y el video están separados más de 750 ms, aumenta "máximo desfase posible"
en la sección de aplausos (el valor inicial ahora es 2000 ms). Antes el programa tenía un
límite fijo de 750 ms y rechazaba una pareja correcta que estuviera más lejos.

LIMITACIONES (para que no te lleves sorpresas)
-----------------------------------------------
- Necesita que el aplauso sea un sonido corto y notablemente más fuerte que lo que lo rodea.
- Necesita que el movimiento de aplaudir sea el movimiento más brusco dentro de la ventana
  analizada (evita mover otras cosas frente a cámara justo antes/después del clap).
- Es una estimación heurística: siempre revisa el resultado escuchando/mirando ese punto
  antes de aplicarlo a un proyecto largo.
"""
