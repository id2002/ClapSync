#!/usr/bin/env python3
"""
clap_sync_gui.py — Version con ventana + terminal. Detecta MULTIPLES aplausos y permite
recortar el cuadro para aislar una sola camara cuando el video es una composicion de 2
camaras lado a lado (ej. hecha en OBS), ademas de un modo diagnostico camara vs camara.

SIN argumentos -> abre la ventana.
CON un argumento (ruta al video) -> funciona como terminal.

REQUISITOS PARA CORRERLO COMO SCRIPT
-------------------------------------
    pip install numpy opencv-python scipy
    (ademas necesitas ffmpeg instalado y en el PATH del sistema)
"""

import argparse
import os
import subprocess
import sys
import tempfile
import threading

import numpy as np


# ----------------------------------------------------------------------
# Logica de analisis
# ----------------------------------------------------------------------

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_audio(video_path, wav_path, start=None, end=None):
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-t", str(end - (start or 0))]
    cmd += ["-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1", wav_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def _robust_threshold(values, sensitivity):
    """Umbral resistente a picos aislados.

    La media y la desviación estándar se desplazan precisamente por los golpes
    que buscamos. La mediana y MAD describen mejor el ruido de fondo.
    """
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    # 1.4826 convierte MAD a una estimación de desviación estándar normal.
    return median + sensitivity * max(1.4826 * mad, 1e-9)


def find_audio_peaks(wav_path, min_gap_s=0.25, sensitivity=3.0, return_strengths=False):
    """sensitivity: cuántas desviaciones estándar debe sobresalir un golpe para contar.
    Más BAJO (ej. 1.5) = detecta más golpes, incluye más ruido de fondo.
    Más ALTO (ej. 5.0) = solo golpes muy claros, ignora ruido, pero puede perderse
    un aplauso si sonó flojo."""
    from scipy.io import wavfile
    from scipy.signal import butter, find_peaks, sosfiltfilt

    sr, data = wavfile.read(wav_path)
    data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Un aplauso tiene un ataque corto y bastante contenido de alta frecuencia.
    # Atenuar graves (voz, ventilador, golpes de mesa) reduce falsos positivos.
    nyquist = sr / 2
    low, high = 500 / nyquist, min(8000 / nyquist, 0.99)
    if low < high:
        filtered = sosfiltfilt(butter(4, [low, high], btype="bandpass", output="sos"), data)
    else:
        filtered = data

    window = max(1, int(sr * 0.005))
    n_windows = len(filtered) // window
    energy = np.array([
        np.sqrt(np.mean(filtered[i * window:(i + 1) * window] ** 2))
        for i in range(n_windows)
    ])
    # El ataque, no el volumen sostenido, es la señal más útil del aplauso.
    rise = np.diff(energy)
    if len(rise) == 0:
        return []

    median_rise = float(np.median(rise))
    spread_rise = max(1.4826 * float(np.median(np.abs(rise - median_rise))), 1e-9)
    threshold = _robust_threshold(rise, sensitivity)
    distance = max(1, int(min_gap_s * sr / window))
    prominence = max(1e-9, threshold - float(np.median(rise)))
    primary_indices, _ = find_peaks(rise, height=threshold, prominence=prominence, distance=distance)

    # Un aplauso que se oye más bajo no debe perderse por completo. Buscamos
    # candidatos secundarios con un umbral menor, pero los marcamos con su
    # fuerza para que el consenso no les dé el mismo peso que a un golpe claro.
    recovery_sensitivity = max(1.0, sensitivity * 0.65)
    recovery_threshold = _robust_threshold(rise, recovery_sensitivity)
    recovery_indices, _ = find_peaks(
        rise,
        height=recovery_threshold,
        prominence=max(spread_rise * 0.75, recovery_threshold - median_rise),
        distance=distance,
    )

    # Unimos ambos pases y mantenemos un único pico por separación mínima.
    candidates = sorted(set(primary_indices).union(recovery_indices), key=lambda i: rise[i], reverse=True)
    selected = []
    for idx in candidates:
        if all(abs(idx - kept) >= distance for kept in selected):
            selected.append(int(idx))
    peak_indices = np.array(sorted(selected), dtype=int)
    coarse_times = [(idx + 1) * window / sr for idx in peak_indices]

    # Segundo pase: cada golpe se detectó con bloques de 10ms (grueso). Ahora que
    # sabemos aproximadamente dónde está cada uno, lo re-analizamos con bloques de
    # 1ms SOLO en una ventanita alrededor, para ubicar el inicio real del golpe
    # con mucha más precisión, en vez de quedarnos con la resolución de 10ms.
    times = [refine_audio_onset(filtered, sr, t) for t in coarse_times]
    if return_strengths:
        strengths = [max(0.0, (rise[idx] - median_rise) / spread_rise) for idx in peak_indices]
        return times, strengths
    return times


def refine_audio_onset(data, sr, coarse_time_s, search_radius_s=0.15, fine_window_s=0.001):
    """Ubica el inicio real de un golpe de audio con precisión fina (~1ms), a partir
    de una estimación gruesa. Busca dónde la energía empieza a subir bruscamente
    cerca del pico, en vez de solo tomar el bloque de 10ms donde cayó el pico."""
    center = int(coarse_time_s * sr)
    radius = int(search_radius_s * sr)
    lo, hi = max(0, center - radius), min(len(data), center + radius)
    segment = data[lo:hi]

    fine_window = max(1, int(sr * fine_window_s))
    n = len(segment) // fine_window
    if n < 3:
        return coarse_time_s  # ventana muy corta para refinar, nos quedamos con la gruesa

    env = np.array([
        np.sqrt(np.mean(segment[i * fine_window:(i + 1) * fine_window] ** 2))
        for i in range(n)
    ])
    peak_idx = int(np.argmax(env))
    peak_val = env[peak_idx]
    baseline = np.median(env[:max(1, peak_idx)]) if peak_idx > 0 else env[0]
    threshold = baseline + 0.2 * (peak_val - baseline)

    onset_idx = 0
    for i in range(peak_idx, -1, -1):
        if env[i] < threshold:
            onset_idx = i + 1
            break

    return (lo + onset_idx * fine_window) / sr


def parse_crop(label):
    """label: 'Todo el cuadro' | 'Izquierda' | 'Derecha' -> (x0,y0,x1,y1) fracciones, o None."""
    if label == "Izquierda":
        return (0.0, 0.0, 0.5, 1.0)
    if label == "Derecha":
        return (0.5, 0.0, 1.0, 1.0)
    if isinstance(label, str) and "," in label:
        try:
            crop = tuple(float(v.strip()) for v in label.split(","))
        except ValueError as exc:
            raise ValueError("El recorte debe ser x0,y0,x1,y1 con valores entre 0 y 1.") from exc
        if len(crop) != 4 or not (0 <= crop[0] < crop[2] <= 1 and 0 <= crop[1] < crop[3] <= 1):
            raise ValueError("El recorte debe cumplir 0 <= x0 < x1 <= 1 y 0 <= y0 < y1 <= 1.")
        return crop
    return None


def find_video_peaks(
    video_path, start=None, end=None, min_gap_s=0.25, crop=None,
    sensitivity=3.0, relaxed=False, return_strengths=False,
):
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
        return ([], fps, []) if return_strengths else ([], fps)

    diffs = np.array(diffs, dtype=np.float64)
    # Solo el modo de aplausos usa un umbral más permisivo. El diagnóstico de
    # cámaras debe conservar su umbral estricto original: allí detectar todo
    # movimiento pequeño empeora el cálculo del retraso entre cámaras.
    visual_sensitivity = max(1.0, sensitivity * 0.65) if relaxed else sensitivity
    threshold = _robust_threshold(diffs, visual_sensitivity)
    distance = max(1, int(min_gap_s * fps))
    prominence = max(1e-9, threshold - float(np.median(diffs)))
    peak_indices, _ = find_peaks(diffs, height=threshold, prominence=prominence, distance=distance)
    times = sorted(start_frame_time + (idx + 1) / fps for idx in peak_indices)
    if return_strengths:
        median_diff = float(np.median(diffs))
        spread_diff = max(1.4826 * float(np.median(np.abs(diffs - median_diff))), 1e-9)
        strengths = [max(0.0, (diffs[idx] - median_diff) / spread_diff) for idx in peak_indices]
        return times, fps, strengths
    return times, fps


def match_peaks(
    a_times, b_times, tolerance=0.75, consensus_tolerance=0.08,
    a_strengths=None, b_strengths=None,
):
    """Empareja a_times (ej. audio) con b_times (ej. video) EN ORDEN CRONOLÓGICO.

    A propósito NO usa 'la pareja más cercana de todas las combinaciones posibles':
    ese enfoque puede dejar que un ruido lejano en el tiempo, que por casualidad cae
    con una distancia menor, le "robe" la pareja al aplauso real (normalmente el
    primero). En su lugar, procesa cada evento de a_times en orden y le busca la
    mejor pareja disponible en b_times SIN retroceder — así el primer evento real
    siempre tiene prioridad para encontrar su pareja correcta.
    """
    # Primero se busca el desfase que se repite. El método anterior elegía el
    # movimiento visual más cercano para cada golpe, aunque fuese una persona
    # moviéndose y no el aplauso. Un desfase A/V real permanece constante.
    if a_strengths is None:
        a_strengths = [1.0] * len(a_times)
    if b_strengths is None:
        b_strengths = [1.0] * len(b_times)
    candidates = [
        (b - a, i, j, max(0.1, float(a_strengths[i])) * max(0.1, float(b_strengths[j])))
        for i, a in enumerate(a_times)
        for j, b in enumerate(b_times)
        if abs(b - a) <= tolerance
    ]
    if not candidates:
        return []

    offsets = np.array([c[0] for c in candidates])
    # Histograma desplazado: evita que el origen de los bins decida un empate.
    bin_width = max(0.02, consensus_tolerance)
    bins = np.round(offsets / bin_width).astype(int)
    unique = np.unique(bins)
    # La cantidad de eventos manda; la energía sólo desempata dos secuencias
    # con el mismo número de coincidencias.
    bin_scores = [
        (int(np.sum(bins == value)), sum(c[3] for c, b in zip(candidates, bins) if b == value))
        for value in unique
    ]
    dominant_bin = unique[max(range(len(unique)), key=lambda k: bin_scores[k])]
    in_bin = offsets[np.abs(offsets - dominant_bin * bin_width) <= bin_width]
    center = float(np.median(in_bin))

    pairs, used_video = [], set()
    for i, a in enumerate(a_times):
        choices = [(abs((b - a) - center), -float(b_strengths[j]), j, b) for j, b in enumerate(b_times)
                   if j not in used_video and abs((b - a) - center) <= consensus_tolerance]
        if choices:
            _, _, j, b = min(choices)
            pairs.append((a, b))
            used_video.add(j)

    # Con un único aplauso no existe consenso: conservamos la pareja más cercana.
    if not pairs:
        _, i, j, _ = min(candidates, key=lambda c: abs(c[0] - center))
        return [(a_times[i], b_times[j])]
    return pairs


def compute_audio_envelope(wav_path, window_s=0.01):
    from scipy.io import wavfile

    sr, data = wavfile.read(wav_path)
    data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)

    window = max(1, int(sr * window_s))
    n = len(data) // window
    if n < 2:
        return np.array([]), np.array([])
    env = np.array([np.sqrt(np.mean(data[i * window:(i + 1) * window] ** 2)) for i in range(n)])
    times = np.arange(n) * window / sr
    return times, env


def compute_video_envelope(video_path, start=None, end=None, crop=None):
    import cv2

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
    times = start_frame_time + (np.arange(1, len(diffs) + 1) / fps)
    return times, np.array(diffs, dtype=np.float64), fps


def cross_correlate_offset(
    audio_times, audio_env, video_times, video_env, max_lag_s=0.3,
    step_s=0.01, center_lag_s=0.0,
):
    """max_lag_s: qué tan lejos busca el desfase, en segundos hacia cada lado.
    OJO: entre más grande, más riesgo de "engancharse" con un patrón repetitivo
    (música, canto, un gesto que se repite) que por casualidad correlaciona fuerte
    en un punto lejano y equivocado. Mantenlo lo más chico posible dado lo que ya
    sabes de tu propio desfase típico — no hace falta buscar más lejos de lo real."""
    """Encuentra el desfase (video_t - audio_t) que mejor alinea los DOS patrones completos
    de actividad, no un solo pico. Más resistente al ruido que el método de aplausos."""
    if len(audio_times) < 2 or len(video_times) < 2:
        raise ValueError("No hay suficiente señal de audio o video en la ventana para correlacionar.")

    grid_start = max(audio_times[0], video_times[0])
    grid_end = min(audio_times[-1], video_times[-1])
    if grid_end - grid_start < 0.5:
        raise ValueError("La ventana es muy corta para este método. Usa al menos 2-3 segundos.")

    grid = np.arange(grid_start, grid_end, step_s)
    a_interp = np.interp(grid, audio_times, audio_env)
    v_interp = np.interp(grid, video_times, video_env)

    a_norm = (a_interp - a_interp.mean()) / (a_interp.std() + 1e-9)
    v_norm = (v_interp - v_interp.mean()) / (v_interp.std() + 1e-9)

    max_shift = max(1, int(max_lag_s / step_s))
    center_shift = int(round(center_lag_s / step_s))
    n_total = len(grid)
    scored = []
    for s in range(center_shift - max_shift, center_shift + max_shift + 1):
        if s >= 0:
            a_seg = a_norm[: n_total - s]
            v_seg = v_norm[s:]
        else:
            a_seg = a_norm[-s:]
            v_seg = v_norm[: n_total + s]
        n = min(len(a_seg), len(v_seg))
        if n < 20:
            continue
        score = float(np.dot(a_seg[:n], v_seg[:n]) / n)
        scored.append((s, score))

    if not scored:
        raise ValueError("La ventana es muy corta para calcular la correlación con confianza.")

    best_shift, best_score = max(scored, key=lambda x: x[1])
    scores_arr = np.array([sc for _, sc in scored])
    confidence = (best_score - scores_arr.mean()) / (scores_arr.std() + 1e-9)
    offset_s = best_shift * step_s
    return offset_s, best_score, confidence


def analyze_correlation(video_path, start=None, end=None, crop=None, max_lag_ms=1000, sensitivity=3.0, log=print):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"No encuentro el archivo: {video_path}")
    if not check_ffmpeg():
        raise EnvironmentError("No encontré ffmpeg. Instálalo y agrégalo al PATH del sistema.")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio.wav")
        log("Extrayendo audio...")
        extract_audio(video_path, wav_path, start, end)

        log("Calculando patrón de energía del audio...")
        raw_audio_times, audio_env = compute_audio_envelope(wav_path)
        audio_times = raw_audio_times + (start or 0)

        # Si existen aplausos claros, dan una referencia temporal mucho más
        # fiable que correlacionar voz/música con el movimiento de todo el
        # cuadro. La correlación se limita alrededor de ese consenso.
        raw_audio_peaks, audio_strengths = find_audio_peaks(
            wav_path, sensitivity=sensitivity, return_strengths=True,
        )
        audio_peaks = [t + (start or 0) for t in raw_audio_peaks]

        log("Calculando patrón de movimiento del video...")
        video_times, video_env, fps = compute_video_envelope(video_path, start, end, crop=crop)

        video_peaks, _, video_strengths = find_video_peaks(
            video_path, start, end, crop=crop, sensitivity=sensitivity, relaxed=True,
            return_strengths=True,
        )

    pairs = match_peaks(
        audio_peaks, video_peaks, tolerance=min(max_lag_ms, 750) / 1000,
        a_strengths=audio_strengths, b_strengths=video_strengths,
    )
    if pairs:
        pair_offsets = np.array([video_t - audio_t for audio_t, video_t in pairs])
        prior_offset_s = float(np.median(pair_offsets))
        # Si estos eventos no coinciden entre sí, no son una referencia segura
        # para la correlación: serían movimientos ajenos al aplauso.
        is_consistent = len(pair_offsets) >= 2 and np.max(np.abs(pair_offsets - prior_offset_s)) <= 0.10
    else:
        is_consistent = False

    if not (pairs and is_consistent):
        raise ValueError(
            "No encontré al menos dos aplausos visuales consistentes. No doy un valor de "
            "correlación porque la actividad global (personas, cámara o música) puede dar un "
            "desfase falso. Selecciona sólo las manos y vuelve a intentarlo."
        )

    # 150 ms permite que la correlación refine, pero no que salte a un ritmo
    # repetitivo distinto del aplauso que acabamos de validar.
    refine_range_s = min(max_lag_ms / 1000, 0.15)
    log(f"Aplausos de referencia: {len(pairs)}; desfase inicial {prior_offset_s * 1000:+.1f} ms.")

    log("Refinando el desfase con los patrones completos de actividad...")
    offset_s, score, confidence = cross_correlate_offset(
        audio_times, audio_env, video_times, video_env, max_lag_s=refine_range_s,
        center_lag_s=prior_offset_s,
    )

    # La correlación usa energía de audio y movimiento de píxeles; es útil para
    # confirmar el consenso, pero no puede desplazar un clap ya validado varios
    # fotogramas por voz, música o movimiento de fondo. Si discrepa más de un
    # frame, se conserva el consenso de los aplausos.
    max_refinement_s = 1.0 / fps
    if abs(offset_s - prior_offset_s) > max_refinement_s:
        log(
            f"La correlación difería {abs(offset_s - prior_offset_s) * fps:.1f} frames del consenso; "
            "se conserva el desfase de aplausos."
        )
        offset_s = prior_offset_s

    return {
        "offset_ms": offset_s * 1000,
        "offset_frames": offset_s * fps,
        "score": score,
        "confidence": confidence,
        "used_clap_prior": bool(pairs and is_consistent),
    }


def format_confidence(confidence):
    base = (
        "Nota: esta confianza mide qué tan definido es el pico encontrado DENTRO del rango de "
        "búsqueda, no si el número en sí es realista. Si el resultado no tiene sentido comparado "
        "con lo que ya sabes de tu equipo, desconfía igual y prueba acotando el rango (± ms).\n\n"
    )
    if confidence > 4:
        return base + "Alta — el patrón coincide claramente en ese desfase, dentro del rango buscado."
    if confidence > 2:
        return base + "Media — coincide, pero no de forma aplastante. Cruza este resultado con otra prueba si puedes."
    return base + ("Baja — la ventana probablemente no tiene suficiente actividad para confiar en este número. "
                    "Prueba con una ventana más larga o con más movimiento/habla.")


def analyze(video_path, start=None, end=None, crop=None, sensitivity=3.0, max_offset_ms=750, log=print):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"No encuentro el archivo: {video_path}")
    if not check_ffmpeg():
        raise EnvironmentError("No encontré ffmpeg. Instálalo y agrégalo al PATH del sistema.")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio.wav")
        log("Extrayendo audio...")
        extract_audio(video_path, wav_path, start, end)

        log("Buscando golpes/aplausos en el audio...")
        raw_audio_peaks, audio_strengths = find_audio_peaks(
            wav_path, sensitivity=sensitivity, return_strengths=True,
        )
        # OJO: extract_audio recorta el audio desde "start" en adelante, así que los
        # tiempos que devuelve find_audio_peaks son relativos a ESE recorte (empiezan en 0),
        # no al video completo. Hay que sumarle "start" para volverlos a tiempo absoluto,
        # igual que ya hace find_video_peaks internamente.
        audio_peaks = [t + (start or 0) for t in raw_audio_peaks]

        log("Buscando golpes/movimientos en el video...")
        video_peaks, fps, video_strengths = find_video_peaks(
            video_path, start, end, crop=crop, sensitivity=sensitivity, relaxed=True,
            return_strengths=True,
        )

    if not audio_peaks or not video_peaks:
        raise ValueError(
            "No detecté suficientes golpes claros. Prueba acotando la ventana (Inicio/Fin), "
            "o cambia el recorte de cámara."
        )

    pairs = match_peaks(
        audio_peaks, video_peaks, tolerance=max_offset_ms / 1000,
        a_strengths=audio_strengths, b_strengths=video_strengths,
    )
    if not pairs:
        audio_list = ", ".join(f"{t:.3f}s" for t in audio_peaks)
        video_list = ", ".join(f"{t:.3f}s" for t in video_peaks)
        raise ValueError(
            f"Detecté golpes por separado, pero ninguno cae dentro del margen configurado "
            f"({max_offset_ms:.0f} ms) para considerarlos el mismo evento.\n\n"
            f"Golpes de AUDIO detectados: {audio_list or '(ninguno)'}\n"
            f"Golpes de VIDEO detectados: {video_list or '(ninguno)'}\n\n"
            "Resta estos tiempos a mano para estimar el desfase real."
        )

    raw_offsets = np.array([video_t - audio_t for audio_t, video_t in pairs])
    consensus_offset_s = float(np.median(raw_offsets))
    results = []
    for audio_t, video_t in pairs:
        raw_offset_s = video_t - audio_t
        results.append({
            "audio_time": audio_t,
            "video_time": video_t,
            "fps": fps,
            # El desfase A/V es fijo: aplicamos el consenso, no una corrección
            # distinta por cada gesto visual candidato.
            "offset_ms": consensus_offset_s * 1000,
            "offset_frames": consensus_offset_s * fps,
            "raw_offset_ms": raw_offset_s * 1000,
        })
    return results


def diagnose_cameras(video_path, start=None, end=None, sensitivity=3.0, log=print):
    log("Analizando cámara IZQUIERDA...")
    left_peaks, fps = find_video_peaks(video_path, start, end, crop=parse_crop("Izquierda"), sensitivity=sensitivity)
    log("Analizando cámara DERECHA...")
    right_peaks, _ = find_video_peaks(video_path, start, end, crop=parse_crop("Derecha"), sensitivity=sensitivity)

    if not left_peaks or not right_peaks:
        raise ValueError("No detecté suficiente movimiento en una de las dos mitades del cuadro.")

    pairs = match_peaks(left_peaks, right_peaks)
    if not pairs:
        raise ValueError("No logré emparejar movimiento entre ambas mitades. Acota Inicio/Fin.")

    results = []
    for lt, rt in pairs:
        offset_ms = (rt - lt) * 1000
        results.append({"left_time": lt, "right_time": rt, "offset_ms": offset_ms, "fps": fps})
    return results


def format_offset_instructions(r):
    offset_ms, offset_frames = r["offset_ms"], r["offset_frames"]
    if abs(offset_ms) < 1:
        return "Prácticamente están sincronizados, no hace falta ajustar nada."
    if offset_ms > 0:
        return (
            f"El video va {offset_ms:.1f} ms POR DELANTE del audio.\n"
            f"  OBS: Sync Offset de esa pista = {int(offset_ms)} ms (positivo).\n"
            f"  DaVinci Resolve: mueve el clip de AUDIO {offset_frames:.2f} frames a la derecha "
            f"(o el de VIDEO esa misma cantidad a la izquierda)."
        )
    return (
        f"El audio va {abs(offset_ms):.1f} ms POR DELANTE del video.\n"
        f"  OBS: Sync Offset de esa pista = {int(offset_ms)} ms (negativo).\n"
        f"  DaVinci Resolve: mueve el clip de AUDIO {abs(offset_frames):.2f} frames a la izquierda "
        f"(o el de VIDEO esa misma cantidad a la derecha)."
    )


# ----------------------------------------------------------------------
# Interfaz grafica
# ----------------------------------------------------------------------

def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("Clap Sync — Detector de desfase audio/video")
    # No asumimos una resolución concreta: en pantallas bajas la interfaz se
    # desplaza y los botones quedan siempre anclados en la parte inferior.
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    initial_width = min(660, max(560, screen_width - 40))
    initial_height = min(790, max(560, screen_height - 80))
    root.geometry(f"{initial_width}x{initial_height}")
    root.minsize(560, 500)
    root.resizable(True, True)

    body = tk.Frame(root)
    body.pack(side="top", fill="both", expand=True)
    canvas = tk.Canvas(body, highlightthickness=0)
    scrollbar = tk.Scrollbar(body, orient="vertical", command=canvas.yview)
    scroll_content = tk.Frame(canvas)
    scroll_window = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def update_scroll_region(event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def resize_scroll_content(event):
        canvas.itemconfigure(scroll_window, width=event.width)

    def scroll_with_wheel(event):
        canvas.yview_scroll(-int(event.delta / 120), "units")

    scroll_content.bind("<Configure>", update_scroll_region)
    canvas.bind("<Configure>", resize_scroll_content)
    canvas.bind("<MouseWheel>", scroll_with_wheel)

    video_path_var = tk.StringVar()
    start_var = tk.StringVar()
    end_var = tk.StringVar()
    crop_var = tk.StringVar(value="Todo el cuadro")
    custom_crop_var = tk.StringVar()
    state = {"results": []}

    def browse_file():
        path = filedialog.askopenfilename(
            title="Selecciona el video de prueba",
            filetypes=[("Videos", "*.mp4 *.mov *.mkv *.avi"), ("Todos los archivos", "*.*")],
        )
        if path:
            video_path_var.set(path)

    frame_top = tk.Frame(scroll_content, padx=15, pady=15)
    frame_top.pack(fill="x")

    tk.Label(frame_top, text="Video:").grid(row=0, column=0, sticky="w")
    tk.Entry(frame_top, textvariable=video_path_var, width=48).grid(row=0, column=1, padx=5)
    tk.Button(frame_top, text="Elegir...", command=browse_file).grid(row=0, column=2)

    tk.Label(frame_top, text="Inicio ventana (s, opcional):").grid(row=1, column=0, sticky="w", pady=(10, 0))
    tk.Entry(frame_top, textvariable=start_var, width=10).grid(row=1, column=1, sticky="w", pady=(10, 0))

    tk.Label(frame_top, text="Fin ventana (s, opcional):").grid(row=2, column=0, sticky="w")
    tk.Entry(frame_top, textvariable=end_var, width=10).grid(row=2, column=1, sticky="w")

    tk.Label(frame_top, text="Cámara a usar (si el video tiene 2 cámaras lado a lado):").grid(
        row=3, column=0, columnspan=2, sticky="w", pady=(10, 0)
    )
    crop_menu = tk.OptionMenu(frame_top, crop_var, "Todo el cuadro", "Izquierda", "Derecha")
    crop_menu.grid(row=4, column=0, sticky="w")

    tk.Label(frame_top, text="Recorte manual opcional (x0,y0,x1,y1; valores 0–1):").grid(
        row=5, column=0, columnspan=2, sticky="w", pady=(6, 0)
    )
    tk.Entry(frame_top, textvariable=custom_crop_var, width=24).grid(row=6, column=0, sticky="w")
    tk.Button(frame_top, text="Seleccionar área de las manos...", command=lambda: choose_manual_crop()).grid(
        row=6, column=1, sticky="w", padx=(8, 0)
    )

    sensitivity_var = tk.DoubleVar(value=3.0)
    tk.Label(frame_top, text="Sensibilidad (baja = detecta más ruido, alta = más estricto):").grid(
        row=7, column=0, columnspan=2, sticky="w", pady=(10, 0)
    )
    tk.Scale(
        frame_top, from_=1.0, to=6.0, resolution=0.5, orient="horizontal",
        variable=sensitivity_var, length=250,
    ).grid(row=8, column=0, columnspan=2, sticky="w")

    clap_max_offset_var = tk.StringVar(value="750")
    tk.Label(
        frame_top,
        text="Solo para 'Analizar (aplausos)': máximo desfase posible (ms):",
    ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(10, 0))
    tk.Entry(frame_top, textvariable=clap_max_offset_var, width=10).grid(row=10, column=0, sticky="w")

    max_lag_var = tk.StringVar(value="1000")
    tk.Label(
        frame_top,
        text="Solo para 'Analizar (correlación)': desfase máximo posible (ms). Con aplausos, refina alrededor de su consenso:",
        wraplength=560, justify="left",
    ).grid(row=11, column=0, columnspan=3, sticky="w", pady=(10, 0))
    tk.Entry(frame_top, textvariable=max_lag_var, width=10).grid(row=12, column=0, sticky="w")

    # Lista de aplausos detectados
    list_frame = tk.Frame(scroll_content, padx=15)
    list_frame.pack(fill="x")
    tk.Label(list_frame, text="Aplausos detectados (doble clic para ver instrucciones):").pack(anchor="w")
    listbox = tk.Listbox(list_frame, height=6, width=85)
    listbox.pack(pady=(0, 5))

    output = tk.Text(scroll_content, height=8, width=80, state="disabled", bg="#f5f5f5")
    output.pack(padx=15, pady=5)

    def log(msg):
        output.configure(state="normal")
        output.insert("end", msg + "\n")
        output.see("end")
        output.configure(state="disabled")
        root.update_idletasks()

    def show_choice(index):
        r = state["results"][index]
        output.configure(state="normal")
        output.delete("1.0", "end")
        output.insert("end", f"--- APLAUSO #{index + 1} ---\n")
        output.insert("end", format_offset_instructions(r))
        output.configure(state="disabled")

    def on_listbox_select(event):
        selection = listbox.curselection()
        if selection:
            show_choice(selection[0])

    listbox.bind("<<ListboxSelect>>", on_listbox_select)

    def get_start_end():
        start = float(start_var.get()) if start_var.get().strip() else None
        end = float(end_var.get()) if end_var.get().strip() else None
        if start is not None and start < 0:
            raise ValueError("El inicio no puede ser negativo.")
        if end is not None and end <= 0:
            raise ValueError("El fin debe ser mayor que cero.")
        if start is not None and end is not None and end <= start:
            raise ValueError("El fin debe ser posterior al inicio.")
        return start, end

    def get_crop():
        # Para aplausos, encuadrar solo las manos evita que el movimiento de
        # rostro, cámara o público se confunda con el evento visual.
        custom = custom_crop_var.get().strip()
        return parse_crop(custom) if custom else parse_crop(crop_var.get())

    def choose_manual_crop():
        """Permite elegir visualmente las manos, sin pedir coordenadas a ojo.

        La diferencia entre fotogramas no sabe qué es un aplauso; limitarla a
        las manos evita que un teclado, un rostro o el fondo se conviertan en
        falsos picos de vídeo.
        """
        video_path = video_path_var.get().strip()
        if not video_path:
            messagebox.showwarning("Falta el video", "Primero elige un archivo de video.")
            return
        try:
            start, _ = get_start_end()
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("No pude abrir el video.")
            if start is not None:
                cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise ValueError("No pude leer un fotograma en el inicio elegido.")

            original_h, original_w = frame.shape[:2]
            # La ventana de selección debe caber incluso en un monitor pequeño.
            scale = min(1.0, 1100 / original_w, 700 / original_h)
            preview = cv2.resize(frame, (round(original_w * scale), round(original_h * scale))) if scale < 1 else frame
            x, y, w, h = cv2.selectROI("Clap Sync — encierra solo las manos y pulsa Enter", preview, False, False)
            cv2.destroyWindow("Clap Sync — encierra solo las manos y pulsa Enter")
            if w <= 0 or h <= 0:
                return  # El usuario canceló con Esc.
            x0, y0 = x / (original_w * scale), y / (original_h * scale)
            x1, y1 = (x + w) / (original_w * scale), (y + h) / (original_h * scale)
            custom_crop_var.set(f"{x0:.4f},{y0:.4f},{x1:.4f},{y1:.4f}")
            log("Área de manos seleccionada. Repite el análisis de aplausos.")
        except Exception as e:
            messagebox.showerror("No pude seleccionar el área", str(e))

    def run_analysis():
        video_path = video_path_var.get().strip()
        if not video_path:
            messagebox.showwarning("Falta el video", "Primero elige un archivo de video.")
            return

        try:
            start, end = get_start_end()
            crop = get_crop()
            max_offset_ms = float(clap_max_offset_var.get())
            if max_offset_ms <= 0:
                raise ValueError("El máximo desfase para aplausos debe ser mayor que cero.")
        except ValueError as e:
            messagebox.showwarning("Valores inválidos", str(e))
            return

        output.configure(state="normal")
        output.delete("1.0", "end")
        output.configure(state="disabled")
        listbox.delete(0, "end")
        analyze_btn.configure(state="disabled", text="Analizando...")

        def worker():
            try:
                results = analyze(
                    video_path, start, end, crop=crop, sensitivity=sensitivity_var.get(),
                    max_offset_ms=max_offset_ms, log=log,
                )
                state["results"] = results
                for i, r in enumerate(results, start=1):
                    listbox.insert(
                        "end",
                        f"{i}. audio={r['audio_time']:.3f}s  video={r['video_time']:.3f}s  "
                        f"consenso={r['offset_ms']:+.1f} ms  (movimiento={r['raw_offset_ms']:+.1f} ms)",
                    )
                log(f"\nDetecté {len(results)} aplauso(s). Si el offset varía mucho entre ellos, "
                    f"probablemente hay emparejamientos falsos o falta elegir la cámara correcta arriba.")
                show_choice(0)
            except Exception as e:
                log(f"\nERROR: {e}")
            finally:
                analyze_btn.configure(state="normal", text="Analizar")

        threading.Thread(target=worker, daemon=True).start()

    def run_diagnose():
        video_path = video_path_var.get().strip()
        if not video_path:
            messagebox.showwarning("Falta el video", "Primero elige un archivo de video.")
            return
        try:
            start, end = get_start_end()
        except ValueError as e:
            messagebox.showwarning("Valores inválidos", str(e))
            return

        output.configure(state="normal")
        output.delete("1.0", "end")
        output.configure(state="disabled")
        listbox.delete(0, "end")
        diagnose_btn.configure(state="disabled", text="Comparando...")

        def worker():
            try:
                results = diagnose_cameras(video_path, start, end, sensitivity=sensitivity_var.get(), log=log)
                log("\n--- IZQUIERDA vs DERECHA ---")
                for i, r in enumerate(results, start=1):
                    log(f"{i}. izquierda={r['left_time']:.3f}s  derecha={r['right_time']:.3f}s  "
                        f"offset={r['offset_ms']:+.1f} ms")
                log("\nEste desfase ya está fijo en el video (no se puede corregir aquí). Úsalo para "
                    "agregar un filtro 'Video Delay (Async)' en OBS a la cámara adelantada, a futuro.")
            except Exception as e:
                log(f"\nERROR: {e}")
            finally:
                diagnose_btn.configure(state="normal", text="Diagnosticar cámaras (izq. vs der.)")

        threading.Thread(target=worker, daemon=True).start()

    def run_correlation():
        video_path = video_path_var.get().strip()
        if not video_path:
            messagebox.showwarning("Falta el video", "Primero elige un archivo de video.")
            return
        try:
            start, end = get_start_end()
            crop = get_crop()
        except ValueError as e:
            messagebox.showwarning("Valores inválidos", str(e))
            return
        try:
            max_lag_ms = float(max_lag_var.get())
        except ValueError:
            messagebox.showwarning("Rango inválido", "El rango de búsqueda (± ms) debe ser un número.")
            return

        output.configure(state="normal")
        output.delete("1.0", "end")
        output.configure(state="disabled")
        listbox.delete(0, "end")
        correlation_btn.configure(state="disabled", text="Correlacionando...")

        def worker():
            try:
                r = analyze_correlation(
                    video_path, start, end, crop=crop, max_lag_ms=max_lag_ms,
                    sensitivity=sensitivity_var.get(), log=log,
                )
                log("\n--- RESULTADO (método de correlación) ---")
                log(f"Confianza: {format_confidence(r['confidence'])}")
                log("")
                log(format_offset_instructions({"offset_ms": r["offset_ms"], "offset_frames": r["offset_frames"]}))
            except Exception as e:
                log(f"\nERROR: {e}")
            finally:
                correlation_btn.configure(state="normal", text="Analizar (correlación, más robusto)")

        threading.Thread(target=worker, daemon=True).start()

    # Este marco no pertenece al área desplazable: permanece visible incluso
    # cuando los controles superiores ocupan más alto que la pantalla.
    btn_frame = tk.Frame(root)
    btn_frame.pack(side="bottom", pady=5)

    analyze_btn = tk.Button(btn_frame, text="Analizar (aplausos)", command=run_analysis, bg="#4a90d9", fg="white", padx=10, pady=5)
    analyze_btn.grid(row=0, column=0, padx=5)

    correlation_btn = tk.Button(
        btn_frame, text="Analizar (correlación, más robusto)", command=run_correlation,
        bg="#2e7d32", fg="white", padx=10, pady=5,
    )
    correlation_btn.grid(row=0, column=1, padx=5)

    diagnose_btn = tk.Button(
        btn_frame, text="Diagnosticar cámaras (izq. vs der.)", command=run_diagnose,
        bg="#888888", fg="white", padx=10, pady=5,
    )
    diagnose_btn.grid(row=1, column=0, columnspan=2, pady=(5, 0))

    root.mainloop()


# ----------------------------------------------------------------------
# Modo terminal
# ----------------------------------------------------------------------

def run_cli():
    parser = argparse.ArgumentParser(description="Detector de desfase audio/video usando aplausos")
    parser.add_argument("video", help="Ruta al video con audio y video juntos")
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--clap", type=int, default=1, help="Qué aplauso usar en orden cronológico (1 = primero)")
    parser.add_argument("--crop", type=str, default=None, help="'left', 'right', o 'x0,y0,x1,y1'")
    parser.add_argument("--diagnose", action="store_true", help="Compara cámara izquierda vs derecha")
    parser.add_argument("--correlation", action="store_true",
                         help="Usa el método de correlación (más robusto al ruido) en vez de detectar aplausos puntuales")
    parser.add_argument("--max-lag", type=float, default=1000,
                        help="Solo con --correlation: desfase máximo posible en ms (default 1000). Con aplausos, se refina alrededor de su consenso.")
    parser.add_argument("--sensitivity", type=float, default=3.0,
                         help="Desviaciones estándar para considerar un golpe (default 3.0). Baja = detecta más ruido, alta = más estricto.")
    parser.add_argument("--max-offset", type=float, default=750,
                        help="Solo con aplausos: máximo desfase A/V posible en ms (default 750). Auméntalo sólo si ya verificaste que el desfase real es mayor.")
    args = parser.parse_args()

    crop_map = {"left": "Izquierda", "right": "Derecha"}
    # Si no es left/right, puede ser el recorte x0,y0,x1,y1 documentado.
    crop = parse_crop(crop_map.get(args.crop, args.crop or "Todo el cuadro"))

    try:
        if args.diagnose:
            results = diagnose_cameras(args.video, args.start, args.end, sensitivity=args.sensitivity)
            print("\n--- IZQUIERDA vs DERECHA ---")
            for i, r in enumerate(results, start=1):
                print(f"{i}. izquierda={r['left_time']:.3f}s  derecha={r['right_time']:.3f}s  offset={r['offset_ms']:+.1f} ms")
            print("\nEste desfase ya está fijo en el video. Úsalo para OBS -> Video Delay (Async).")
            return

        if args.correlation:
            r = analyze_correlation(
                args.video, args.start, args.end, crop=crop, max_lag_ms=args.max_lag,
                sensitivity=args.sensitivity,
            )
            print("\n--- RESULTADO (método de correlación) ---")
            print(f"Confianza: {format_confidence(r['confidence'])}\n")
            print(format_offset_instructions({"offset_ms": r["offset_ms"], "offset_frames": r["offset_frames"]}))
            return

        results = analyze(
            args.video, args.start, args.end, crop=crop, sensitivity=args.sensitivity,
            max_offset_ms=args.max_offset,
        )
    except Exception as e:
        sys.exit(f"ERROR: {e}")

    print("\nGolpes/aplausos detectados y emparejados:")
    for i, r in enumerate(results, start=1):
        print(
            f"  {i}. audio={r['audio_time']:.3f}s  video={r['video_time']:.3f}s  "
            f"consenso={r['offset_ms']:+.1f} ms  (movimiento={r['raw_offset_ms']:+.1f} ms)"
        )

    if args.clap < 1 or args.clap > len(results):
        sys.exit(f"\n--clap {args.clap} no existe. Usa un número entre 1 y {len(results)}.")

    chosen = results[args.clap - 1]
    print(f"\n--- USANDO APLAUSO #{args.clap} ---")
    print(format_offset_instructions(chosen))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli()
    else:
        launch_gui()
