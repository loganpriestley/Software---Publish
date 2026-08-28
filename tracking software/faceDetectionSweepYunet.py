
import cv2
import time
import itertools
import csv
import os
 
# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
 
#the r in front allows python to process the path as intended
VIDEO_PATH = r"C:\Users\33log\OneDrive\Desktop\TrackingCamera\Software\SweepTestFootage.mp4"
MODEL_PATH = "face_detection_yunet_2026may.onnx"   # path to the downloaded YuNet model
 
# lists of values to sweep -- every combination of these will be tested
SCORE_THRESHOLD_VALUES = [0.6, 0.7, 0.8, 0.9]
NMS_THRESHOLD_VALUES = [0.3, 0.4, 0.5]
 
# held fixed across the sweep
TOP_K = 5000   # rarely matters for a single-face rig, so not swept by default
 
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_results_yunet.csv")
 
# ---------------------------------------------------------------------------
 
 
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
 
 
def pick_largest(faces):
    if faces is None or len(faces) == 0:
        return None
    largest = faces[0]
    for face in faces:
        if largest[2] * largest[3] < face[2] * face[3]:
            largest = face
    return largest
 
 
def run_combo(detector, frames, score_threshold, nms_threshold, top_k):
    # thresholds can be changed on the existing detector without recreating it,
    # which keeps the sweep fast
    detector.setScoreThreshold(score_threshold)
    detector.setNMSThreshold(nms_threshold)
    detector.setTopK(top_k)
 
    detected_frames = 0
    total_candidates = 0  # raw candidate count before picking largest -- false-positive-pressure indicator
 
    start = time.time()
    for frame in frames:
        _, faces = detector.detect(frame)
        n_candidates = 0 if faces is None else len(faces)
        total_candidates += n_candidates
        if pick_largest(faces) is not None:
            detected_frames += 1
    elapsed = time.time() - start
 
    total_frames = len(frames)
    return {
        "score_threshold": score_threshold,
        "nms_threshold": nms_threshold,
        "top_k": top_k,
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "detection_rate": detected_frames / total_frames if total_frames else 0,
        "avg_candidates_per_frame": total_candidates / total_frames if total_frames else 0,
        "total_time_s": elapsed,
        "avg_ms_per_frame": (elapsed / total_frames * 1000) if total_frames else 0,
    }
 
 
def print_table(results):
    headers = ["score_thr", "nms_thr", "top_k", "detected/total", "rate", "avg cand/frame", "ms/frame"]
    rows = []
    for r in results:
        rows.append([
            f"{r['score_threshold']}",
            f"{r['nms_threshold']}",
            f"{r['top_k']}",
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
    for row, r in sorted(zip(rows, results), key=lambda pair: pair[1]["detection_rate"], reverse=True):
        print(fmt_row(row))
 
 
def main():
    print(f"Loading frames from {VIDEO_PATH}...")
    frames = load_frames(VIDEO_PATH)
    print(f"Loaded {len(frames)} frames.\n")
 
    frame_h, frame_w = frames[0].shape[:2]
    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH, "", (frame_w, frame_h),
        SCORE_THRESHOLD_VALUES[0], NMS_THRESHOLD_VALUES[0], TOP_K
    )
 
    combos = list(itertools.product(SCORE_THRESHOLD_VALUES, NMS_THRESHOLD_VALUES))
    print(f"Running {len(combos)} parameter combinations...\n")
 
    results = []
    for score_threshold, nms_threshold in combos:
        r = run_combo(detector, frames, score_threshold, nms_threshold, TOP_K)
        results.append(r)
        print(f"  done: score_threshold={score_threshold} nms_threshold={nms_threshold} -> "
              f"{r['detected_frames']}/{r['total_frames']} ({r['detection_rate']:.1%})")
 
    print("\n--- Results (sorted by detection rate) ---\n")
    print_table(results)
 
    # order best-to-worst: highest detection rate first, then lower
    # avg_candidates_per_frame as the tiebreaker (fewer false-positive candidates)
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