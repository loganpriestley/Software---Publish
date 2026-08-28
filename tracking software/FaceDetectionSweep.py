import cv2
import time
import itertools
import csv
 

#the  r in front allows python to process the path as intended
VIDEO_PATH = r"C:\Users\33log\OneDrive\Desktop\TrackingCamera\Software\SweepTestFootage.mp4"
 
# lists of values to sweep
PROCESS_FAC_VALUES = [1.05, 1.1, 1.2, 1.3]
NEIGHBORS_VALUES = [3, 4, 5, 6]
SCALE_FACTOR = [0.3,0.4,0.5,0.6,0.75]
EQUALIZE_HIST_VALUES = [False]
 
# held fixed across the sweep 

MIN_FACE_SIZE = (60, 60)
 
OUTPUT_CSV = "sweep_results.csv"   # set to None to skip saving
 
# ---------------------------------------------------------------------------
 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
def load_frames(video_path):
    """Reads every frame of the video into memory once so each sweep combo runs against identical frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
 
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames
 
 
def detect_face(frame, scale_factor, process_fac, neighbors, min_face_size,equalize):
    small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if equalize:
        gray = cv2.equalizeHist(gray)
 
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=process_fac,
        minNeighbors=neighbors,
        minSize=(int(min_face_size[0] * scale_factor), int(min_face_size[1] * scale_factor))
    )
 
    if len(faces) == 0:
        return None, len(faces)
 
    largest = faces[0]
    for face in faces:
        if largest[2] * largest[3] < face[2] * face[3]:
            largest = face
 
    return largest, len(faces)
 
 
def run_combo(frames, scale_factor, process_fac, neighbors, min_face_size, equalize):
    detected_frames = 0
    total_candidates = 0  # sum of all raw detections before picking largest, across frames -- a rough false-positive-pressure indicator
 
    start = time.time()
    for frame in frames:
        result, n_candidates = detect_face(frame, scale_factor, process_fac, neighbors, min_face_size, equalize)
        total_candidates += n_candidates
        if result is not None:
            detected_frames += 1
    elapsed = time.time() - start
 
    total_frames = len(frames)
    return {
        "process_fac": process_fac,
        "neighbors": neighbors,
        "scale_factor": scale_factor,
        "equalize_hist": equalize,
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "detection_rate": detected_frames / total_frames if total_frames else 0,
        "avg_candidates_per_frame": total_candidates / total_frames if total_frames else 0,
        "total_time_s": elapsed,
        "avg_ms_per_frame": (elapsed / total_frames * 1000) if total_frames else 0,
    }
 
 
def print_table(results):
    headers = ["process_fac", "neighbors", "scale_factor", "equalize", "detected/total", "rate", "avg cand/frame", "ms/frame"]
    rows = []
    for r in results:
        rows.append([
            f"{r['process_fac']}",
            f"{r['neighbors']}",
            f"{r['scale_factor']}",
            f"{r['equalize_hist']}",
            f"{r['detected_frames']}/{r['total_frames']}",
            f"{r['detection_rate']:.1%}",
            f"{r['avg_candidates_per_frame']:.2f}",
            f"{r['avg_ms_per_frame']:.1f}",
        ])
 
    col_widths = [max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
 
    def fmt_row(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, col_widths))
 
    print(fmt_row(headers))
    print("  ".join("-" * w for w in col_widths))
    # sort by detection rate descending so the best candidates are on top
    for row, r in sorted(zip(rows, results), key=lambda pair: pair[1]["detection_rate"], reverse=True):
        print(fmt_row(row))
 
 
def main():
    print(f"Loading frames from {VIDEO_PATH}...")
    frames = load_frames(VIDEO_PATH)
    print(f"Loaded {len(frames)} frames.\n")
 
    combos = list(itertools.product(PROCESS_FAC_VALUES, NEIGHBORS_VALUES,SCALE_FACTOR,EQUALIZE_HIST_VALUES))
    print(f"Running {len(combos)} parameter combinations...\n")
 
    results = []
    for process_fac, neighbors, scale_factor, equalize_hist_value in combos:
        r = run_combo(frames, scale_factor, process_fac, neighbors, MIN_FACE_SIZE, equalize_hist_value)
        results.append(r)
        print(f"  done: process_fac={process_fac} neighbors={neighbors} scale_factor = {scale_factor} equalize_hist_value = {equalize_hist_value}-> "
              f"{r['detected_frames']}/{r['total_frames']} ({r['detection_rate']:.1%})")
 
    print("\n--- Results (sorted by detection rate) ---\n")
    print_table(results)
 
    # order best-to-worst: highest detection rate first, and for ties,
    # prefer the lower avg_candidates_per_frame (fewer false-positive candidates)
    results_sorted = sorted(
        results,
        key=lambda r: (-r["detection_rate"], r["avg_candidates_per_frame"])
    )

    if OUTPUT_CSV:
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results_sorted[0].keys())
            writer.writeheader()
            writer.writerows(results_sorted)
        print(f"\nFull results saved to {OUTPUT_CSV} (ordered best to worst)")

if __name__ == "__main__":
    main()