#!/usr/bin/env python3
"""
clap_sync.py — Detecta el desfase de audio/video usando aplausos. Ahora soporta
RECORTAR el cuadro (--crop) para casos donde el video es una composición de 2
cámaras lado a lado (ej. una escena dividida hecha en OBS), y un modo diagnóstico
(--diagnose) para medir cuánto se adelanta una cámara respecto a la otra.

IMPORTANTE si tu video es una composición de 2 cámaras ya combinadas por OBS:
- El desfase ENTRE cámaras (una adelantada respecto a otra) ya quedó fijo en los
  píxeles del archivo — no se puede deshacer en post. Usa --diagnose para medir
  cuánto es, y corrígelo A FUTURO en OBS con un filtro "Video Delay (Async)" en
  la fuente de cámara que va adelantada.
- El desfase de AUDIO sí se puede corregir en post. Usa --crop para elegir UNA
  sola cámara como referencia (la que más te importa, ej. donde se ve hablar a
  alguien) en vez de mezclar el movimiento de ambas cámaras a la vez.

REQUISITOS
----------
    pip install numpy opencv-python scipy --break-system-packages
    (además necesitas ffmpeg instalado y accesible desde la terminal)

USO
---
    # Uso normal (un solo video/cámara en el cuadro completo)
    python clap_sync.py video.mp4

    # Video con 2 cámaras lado a lado: usar solo la de la izquierda como referencia
    python clap_sync.py video.mp4 --crop left

    # Medir cuánto se adelanta una cámara respecto a la otra (diagnóstico, para OBS)
    python clap_sync.py video.mp4 --diagnose

    # Recorte personalizado: x0,y0,x1,y1 como fracciones de 0 a 1 del cuadro completo
    python clap_sync.py video.mp4 --crop 0.0,0.0,0.5,1.0
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("No encontré ffmpeg. Instálalo y asegúrate de que el comando 'ffmpeg' funcione en tu terminal.")


def extract_audio(video_path, wav_path, start=None, end=None):
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-t", str(end - (start or 0))]
    cmd += ["-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1", wav_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def find_audio_peaks(wav_path, min_gap_s=0.25):
    from scipy.io import wavfile
    from scipy.signal import find_peaks

    sr, data = wavfile.read(wav_path)
    data = data.astype(np.float64)

    window = max(1, int(sr * 0.010))
    n_windows = len(data) // window
    energy = np.array([
        np.sqrt(np.mean(data[i * window:(i + 1) * window] ** 2))
        for i in range(n_windows)
    ])
    rise = np.diff(energy)
    if len(rise) == 0:
        return []

    threshold = np.mean(rise) + 3 * np.std(rise)
    distance = max(1, int(min_gap_s * sr / window))
    peak_indices, _ = find_peaks(rise, height=threshold, distance=distance)
    return sorted((idx + 1) * window / sr for idx in peak_indices)


def parse_crop(crop_arg):
    """Devuelve (x0,y0,x1,y1) como fracciones 0-1, o None si no hay recorte."""
    if crop_arg is None:
        return None
    if crop_arg == "left":
        return (0.0, 0.0, 0.5, 1.0)
    if crop_arg == "right":
        return (0.5, 0.0, 1.0, 1.0)
    try:
        parts = [float(x) for x in crop_arg.split(",")]
        if len(parts) != 4:
            raise ValueError
        x0, y0, x1, y1 = parts
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError
        return (x0, y0, x1, y1)
    except ValueError:
        sys.exit(f"--crop inválido: '{crop_arg}'. Usa 'left', 'right', o 'x0,y0,x1,y1' (fracciones 0-1).")


def find_video_peaks(video_path, start=None, end=None, min_gap_s=0.25, crop=None):
    import cv2
    from scipy.signal import find_peaks

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No pude abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if crop is not None:
        x0, y0, x1, y1 = crop
        px0, py0, px1, py1 = int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)
    else:
        px0, py0, px1, py1 = 0, 0, width, height

    if start:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    prev_gray = None
    diffs = []
    frame_index = 0
    start_frame_time = start or 0

    while True:
        if end is not None and (start_frame_time + frame_index / fps) > end:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[py0:py1, px0:px1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))
        if prev_gray is not None:
            diffs.append(np.sum(np.abs(gray.astype(np.int32) - prev_gray.astype(np.int32))))
        prev_gray = gray
        frame_index += 1

    cap.release()
    if not diffs:
        return [], fps

    diffs = np.array(diffs, dtype=np.float64)
    threshold = np.mean(diffs) + 3 * np.std(diffs)
    distance = max(1, int(min_gap_s * fps))
    peak_indices, _ = find_peaks(diffs, height=threshold, distance=distance)
    times = sorted(start_frame_time + (idx + 1) / fps for idx in peak_indices)
    return times, fps


def match_peaks(a_times, b_times, tolerance=0.75):
    candidates = []
    for ai, a in enumerate(a_times):
        for bi, b in enumerate(b_times):
            d = abs(b - a)
            if d <= tolerance:
                candidates.append((d, ai, bi))
    candidates.sort(key=lambda c: c[0])

    used_a, used_b, pairs = set(), set(), []
    for d, ai, bi in candidates:
        if ai in used_a or bi in used_b:
            continue
        pairs.append((a_times[ai], b_times[bi]))
        used_a.add(ai)
        used_b.add(bi)
    pairs.sort(key=lambda p: p[0])
    return pairs


def print_offset(label, offset_ms, fps):
    offset_frames = offset_ms / 1000 * fps
    if abs(offset_ms) < 1:
        print(f"{label}: ya sincronizados.")
    elif offset_ms > 0:
        print(f"{label}: el segundo va {offset_ms:.1f} ms POR DELANTE del primero ({offset_frames:.2f} frames).")
    else:
        print(f"{label}: el primero va {abs(offset_ms):.1f} ms POR DELANTE del segundo ({abs(offset_frames):.2f} frames).")


def run_diagnose(video_path, start, end):
    """Compara movimiento del lado izquierdo vs derecho del cuadro (2 cámaras compuestas)."""
    print("Analizando cámara IZQUIERDA del cuadro...")
    left_peaks, fps = find_video_peaks(video_path, start, end, crop=parse_crop("left"))
    print("Analizando cámara DERECHA del cuadro...")
    right_peaks, _ = find_video_peaks(video_path, start, end, crop=parse_crop("right"))

    if not left_peaks or not right_peaks:
        sys.exit(
            "No detecté suficiente movimiento en una de las dos mitades. Prueba acotando "
            "la ventana con --start/--end justo alrededor del aplauso."
        )

    pairs = match_peaks(left_peaks, right_peaks)
    if not pairs:
        sys.exit("No logré emparejar movimiento entre ambas mitades. Acota la ventana con --start/--end.")

    print(f"\nEmparejamientos encontrados: {len(pairs)}")
    for i, (lt, rt) in enumerate(pairs, start=1):
        offset_ms = (rt - lt) * 1000
        print(f"\n  #{i}: izquierda={lt:.3f}s  derecha={rt:.3f}s")
        print_offset("  Izquierda vs Derecha", offset_ms, fps)

    print(
        "\nRecuerda: esto NO se puede corregir en este archivo (ya está soldado en los "
        "píxeles). Usa este número para agregar un filtro 'Video Delay (Async)' a la cámara "
        "adelantada en OBS, de cara a tu próxima grabación."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Ruta al video con audio y video juntos")
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--clap", type=int, default=1, help="Qué aplauso usar en orden cronológico (1 = primero)")
    parser.add_argument("--crop", type=str, default=None, help="'left', 'right', o 'x0,y0,x1,y1' (fracciones 0-1)")
    parser.add_argument("--diagnose", action="store_true", help="Compara cámara izquierda vs derecha (no usa audio)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"No encuentro el archivo: {args.video}")
    check_ffmpeg()

    if args.diagnose:
        run_diagnose(args.video, args.start, args.end)
        return

    crop = parse_crop(args.crop)

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio.wav")
        print("Extrayendo audio...")
        extract_audio(args.video, wav_path, args.start, args.end)

        print("Buscando golpes/aplausos en el audio...")
        raw_audio_peaks = find_audio_peaks(wav_path)
        # extract_audio recorta el audio desde "start" en adelante; find_audio_peaks
        # devuelve tiempos relativos a ese recorte. Hay que sumar "start" para volver
        # a tiempo absoluto, igual que ya hace find_video_peaks.
        audio_peaks = [t + (args.start or 0) for t in raw_audio_peaks]

        print("Buscando golpes/movimientos en el video" + (f" (recorte: {args.crop})" if args.crop else "") + "...")
        video_peaks, fps = find_video_peaks(args.video, args.start, args.end, crop=crop)

    if not audio_peaks or not video_peaks:
        sys.exit(
            "No detecté suficientes golpes claros. Prueba acotando la ventana con --start/--end, "
            "o usando --crop para aislar una sola cámara."
        )

    pairs = match_peaks(audio_peaks, video_peaks)
    if not pairs:
        sys.exit(
            "Detecté golpes en audio y video pero no logré emparejarlos. Prueba --start/--end "
            "más ajustado, o --crop para aislar la cámara correcta."
        )

    print("\nAplausos detectados y emparejados:")
    for i, (a, v) in enumerate(pairs, start=1):
        offset_ms = (v - a) * 1000
        print(f"  {i}. audio={a:.3f}s  video={v:.3f}s  offset={offset_ms:+.1f} ms")

    if args.clap < 1 or args.clap > len(pairs):
        sys.exit(f"\n--clap {args.clap} no existe. Usa un número entre 1 y {len(pairs)}.")

    a, v = pairs[args.clap - 1]
    offset_ms = (v - a) * 1000
    print(f"\n--- USANDO APLAUSO #{args.clap} ---")
    print_offset("Video vs Audio", offset_ms, fps)


if __name__ == "__main__":
    main()
