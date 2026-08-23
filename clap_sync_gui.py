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


def find_audio_peaks(wav_path, min_gap_s=0.25, sensitivity=3.0):
    """sensitivity: cuántas desviaciones estándar debe sobresalir un golpe para contar.
    Más BAJO (ej. 1.5) = detecta más golpes, incluye más ruido de fondo.
    Más ALTO (ej. 5.0) = solo golpes muy claros, ignora ruido, pero puede perderse
    un aplauso si sonó flojo."""
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

    threshold = np.mean(rise) + sensitivity * np.std(rise)
    distance = max(1, int(min_gap_s * sr / window))
    peak_indices, _ = find_peaks(rise, height=threshold, distance=distance)
    return sorted((idx + 1) * window / sr for idx in peak_indices)


def parse_crop(label):
    """label: 'Todo el cuadro' | 'Izquierda' | 'Derecha' -> (x0,y0,x1,y1) fracciones, o None."""
    if label == "Izquierda":
        return (0.0, 0.0, 0.5, 1.0)
    if label == "Derecha":
        return (0.5, 0.0, 1.0, 1.0)
    return None


def find_video_peaks(video_path, start=None, end=None, min_gap_s=0.25, crop=None, sensitivity=3.0):
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
    threshold = np.mean(diffs) + sensitivity * np.std(diffs)
    distance = max(1, int(min_gap_s * fps))
    peak_indices, _ = find_peaks(diffs, height=threshold, distance=distance)
    times = sorted(start_frame_time + (idx + 1) / fps for idx in peak_indices)
    return times, fps


def match_peaks(a_times, b_times, tolerance=0.75):
    """Empareja a_times (ej. audio) con b_times (ej. video) EN ORDEN CRONOLÓGICO.

    A propósito NO usa 'la pareja más cercana de todas las combinaciones posibles':
    ese enfoque puede dejar que un ruido lejano en el tiempo, que por casualidad cae
    con una distancia menor, le "robe" la pareja al aplauso real (normalmente el
    primero). En su lugar, procesa cada evento de a_times en orden y le busca la
    mejor pareja disponible en b_times SIN retroceder — así el primer evento real
    siempre tiene prioridad para encontrar su pareja correcta.
    """
    pairs = []
    start_j = 0
    for a in a_times:
        best_j, best_d = None, None
        for j in range(start_j, len(b_times)):
            d = b_times[j] - a
            if d > tolerance:
                break  # b_times está ordenado; más adelante solo se aleja más
            if d < -tolerance:
                continue  # todavía no llegamos a la zona de tolerancia
            ad = abs(d)
            if best_d is None or ad < best_d:
                best_d, best_j = ad, j
        if best_j is not None:
            pairs.append((a, b_times[best_j]))
            start_j = best_j + 1
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


def cross_correlate_offset(audio_times, audio_env, video_times, video_env, max_lag_s=0.3, step_s=0.01):
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
    n_total = len(grid)
    scored = []
    for s in range(-max_shift, max_shift + 1):
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


def analyze_correlation(video_path, start=None, end=None, crop=None, max_lag_ms=300, log=print):
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

        log("Calculando patrón de movimiento del video...")
        video_times, video_env, fps = compute_video_envelope(video_path, start, end, crop=crop)

    log(f"Buscando el desfase (dentro de ±{max_lag_ms:.0f} ms) que mejor alinea ambos patrones...")
    offset_s, score, confidence = cross_correlate_offset(
        audio_times, audio_env, video_times, video_env, max_lag_s=max_lag_ms / 1000
    )

    return {"offset_ms": offset_s * 1000, "offset_frames": offset_s * fps, "score": score, "confidence": confidence}


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


def analyze(video_path, start=None, end=None, crop=None, sensitivity=3.0, log=print):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"No encuentro el archivo: {video_path}")
    if not check_ffmpeg():
        raise EnvironmentError("No encontré ffmpeg. Instálalo y agrégalo al PATH del sistema.")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio.wav")
        log("Extrayendo audio...")
        extract_audio(video_path, wav_path, start, end)

        log("Buscando golpes/aplausos en el audio...")
        raw_audio_peaks = find_audio_peaks(wav_path, sensitivity=sensitivity)
        # OJO: extract_audio recorta el audio desde "start" en adelante, así que los
        # tiempos que devuelve find_audio_peaks son relativos a ESE recorte (empiezan en 0),
        # no al video completo. Hay que sumarle "start" para volverlos a tiempo absoluto,
        # igual que ya hace find_video_peaks internamente.
        audio_peaks = [t + (start or 0) for t in raw_audio_peaks]

        log("Buscando golpes/movimientos en el video...")
        video_peaks, fps = find_video_peaks(video_path, start, end, crop=crop, sensitivity=sensitivity)

    if not audio_peaks or not video_peaks:
        raise ValueError(
            "No detecté suficientes golpes claros. Prueba acotando la ventana (Inicio/Fin), "
            "o cambia el recorte de cámara."
        )

    pairs = match_peaks(audio_peaks, video_peaks)
    if not pairs:
        audio_list = ", ".join(f"{t:.3f}s" for t in audio_peaks)
        video_list = ", ".join(f"{t:.3f}s" for t in video_peaks)
        raise ValueError(
            "Detecté golpes por separado, pero ninguno cae dentro del margen normal (0.75s) "
            "para considerarlos el mismo evento. Es probable que el desfase real sea MAYOR "
            "a 0.75 segundos.\n\n"
            f"Golpes de AUDIO detectados: {audio_list or '(ninguno)'}\n"
            f"Golpes de VIDEO detectados: {video_list or '(ninguno)'}\n\n"
            "Resta estos tiempos a mano para estimar el desfase real."
        )

    results = []
    for audio_t, video_t in pairs:
        offset_s = video_t - audio_t
        results.append({
            "audio_time": audio_t,
            "video_time": video_t,
            "fps": fps,
            "offset_ms": offset_s * 1000,
            "offset_frames": offset_s * fps,
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
    root.geometry("660x690")
    root.resizable(False, False)

    video_path_var = tk.StringVar()
    start_var = tk.StringVar()
    end_var = tk.StringVar()
    crop_var = tk.StringVar(value="Todo el cuadro")
    state = {"results": []}

    def browse_file():
        path = filedialog.askopenfilename(
            title="Selecciona el video de prueba",
            filetypes=[("Videos", "*.mp4 *.mov *.mkv *.avi"), ("Todos los archivos", "*.*")],
        )
        if path:
            video_path_var.set(path)

    frame_top = tk.Frame(root, padx=15, pady=15)
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

    sensitivity_var = tk.DoubleVar(value=3.0)
    tk.Label(frame_top, text="Sensibilidad (baja = detecta más ruido, alta = más estricto):").grid(
        row=5, column=0, columnspan=2, sticky="w", pady=(10, 0)
    )
    tk.Scale(
        frame_top, from_=1.0, to=6.0, resolution=0.5, orient="horizontal",
        variable=sensitivity_var, length=250,
    ).grid(row=6, column=0, columnspan=2, sticky="w")

    max_lag_var = tk.StringVar(value="300")
    tk.Label(
        frame_top,
        text="Solo para 'Analizar (correlación)': rango de búsqueda ± ms (déjalo chico, evita enganchar música/ritmo):",
        wraplength=560, justify="left",
    ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(10, 0))
    tk.Entry(frame_top, textvariable=max_lag_var, width=10).grid(row=8, column=0, sticky="w")

    # Lista de aplausos detectados
    list_frame = tk.Frame(root, padx=15)
    list_frame.pack(fill="x")
    tk.Label(list_frame, text="Aplausos detectados (doble clic para ver instrucciones):").pack(anchor="w")
    listbox = tk.Listbox(list_frame, height=6, width=85)
    listbox.pack(pady=(0, 5))

    output = tk.Text(root, height=8, width=80, state="disabled", bg="#f5f5f5")
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
        return start, end

    def run_analysis():
        video_path = video_path_var.get().strip()
        if not video_path:
            messagebox.showwarning("Falta el video", "Primero elige un archivo de video.")
            return

        start, end = get_start_end()
        crop = parse_crop(crop_var.get())

        output.configure(state="normal")
        output.delete("1.0", "end")
        output.configure(state="disabled")
        listbox.delete(0, "end")
        analyze_btn.configure(state="disabled", text="Analizando...")

        def worker():
            try:
                results = analyze(video_path, start, end, crop=crop, sensitivity=sensitivity_var.get(), log=log)
                state["results"] = results
                for i, r in enumerate(results, start=1):
                    listbox.insert(
                        "end",
                        f"{i}. audio={r['audio_time']:.3f}s  video={r['video_time']:.3f}s  "
                        f"offset={r['offset_ms']:+.1f} ms",
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
        start, end = get_start_end()

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
        start, end = get_start_end()
        crop = parse_crop(crop_var.get())
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
                r = analyze_correlation(video_path, start, end, crop=crop, max_lag_ms=max_lag_ms, log=log)
                log("\n--- RESULTADO (método de correlación) ---")
                log(f"Confianza: {format_confidence(r['confidence'])}")
                log("")
                log(format_offset_instructions({"offset_ms": r["offset_ms"], "offset_frames": r["offset_frames"]}))
            except Exception as e:
                log(f"\nERROR: {e}")
            finally:
                correlation_btn.configure(state="normal", text="Analizar (correlación, más robusto)")

        threading.Thread(target=worker, daemon=True).start()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=5)

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
    parser.add_argument("--max-lag", type=float, default=300,
                         help="Solo con --correlation: rango de búsqueda en ± ms (default 300). Más chico = menos riesgo de engancharse con música/ritmo.")
    parser.add_argument("--sensitivity", type=float, default=3.0,
                         help="Desviaciones estándar para considerar un golpe (default 3.0). Baja = detecta más ruido, alta = más estricto.")
    args = parser.parse_args()

    crop_map = {"left": "Izquierda", "right": "Derecha"}
    crop = parse_crop(crop_map.get(args.crop, "Todo el cuadro"))

    try:
        if args.diagnose:
            results = diagnose_cameras(args.video, args.start, args.end, sensitivity=args.sensitivity)
            print("\n--- IZQUIERDA vs DERECHA ---")
            for i, r in enumerate(results, start=1):
                print(f"{i}. izquierda={r['left_time']:.3f}s  derecha={r['right_time']:.3f}s  offset={r['offset_ms']:+.1f} ms")
            print("\nEste desfase ya está fijo en el video. Úsalo para OBS -> Video Delay (Async).")
            return

        if args.correlation:
            r = analyze_correlation(args.video, args.start, args.end, crop=crop, max_lag_ms=args.max_lag)
            print("\n--- RESULTADO (método de correlación) ---")
            print(f"Confianza: {format_confidence(r['confidence'])}\n")
            print(format_offset_instructions({"offset_ms": r["offset_ms"], "offset_frames": r["offset_frames"]}))
            return

        results = analyze(args.video, args.start, args.end, crop=crop, sensitivity=args.sensitivity)
    except Exception as e:
        sys.exit(f"ERROR: {e}")

    print("\nGolpes/aplausos detectados y emparejados:")
    for i, r in enumerate(results, start=1):
        print(f"  {i}. audio={r['audio_time']:.3f}s  video={r['video_time']:.3f}s  offset={r['offset_ms']:+.1f} ms")

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