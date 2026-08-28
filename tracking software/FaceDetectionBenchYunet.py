"""
Detection benchmark harness -- YuNet version.
 
Runs YuNet face detection against a prerecorded video file, draws the
bounding box + landmarks on screen so you can visually confirm what it's
detecting, and reports a detection rate summary at the end.
 
Before running this you need the YuNet model file downloaded once:
https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
Save it somewhere and point MODEL_PATH at it below.
 
Controls while running:
    SPACE = pause / resume
    N     = step one frame forward while paused
    Q     = quit early (still prints summary for frames seen so far)
"""
 
import cv2
import time
 
# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
 
VIDEO_PATH = r"C:\Users\33log\OneDrive\Desktop\TrackingCamera\Software\SweepTestFootage.mp4"           # path to your prerecorded test video
MODEL_PATH = "face_detection_yunet_2026may.onnx"   # path to the downloaded YuNet model
 
# YuNet hyperparameters -- change these between runs to compare settings
SCORE_THRESHOLD = 0.9   # confidence cutoff -- higher = stricter, fewer false positives
NMS_THRESHOLD = 0.3     # overlap merging threshold, rarely needs tuning
TOP_K = 5000            # max candidates considered before filtering
 
# optional: save an annotated copy of the video so you can review it later
SAVE_OUTPUT_PATH = "annotated_output.mp4"   # or None
 
# ---------------------------------------------------------------------------
 
 
def pick_largest(faces):
    """faces is an (N, 15) array: [x, y, w, h, 5x(landmark x,y), confidence].
    Returns the single row with the largest box area, or None if empty."""
    if faces is None or len(faces) == 0:
        return None
    largest = faces[0]
    for face in faces:
        if largest[2] * largest[3] < face[2] * face[3]:
            largest = face
    return largest
 
 
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Could not open video file: {VIDEO_PATH}")
        return
 
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay_ms = int(1000 / fps)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH, "", (frame_w, frame_h),
        SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
    )
 
    writer = None
    if SAVE_OUTPUT_PATH:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(SAVE_OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
 
    total_frames = 0
    detected_frames = 0
    paused = False
 
    print("Running benchmark (YuNet)...")
    print(f"  video: {VIDEO_PATH}")
    print(f"  score_threshold={SCORE_THRESHOLD}  nms_threshold={NMS_THRESHOLD}  top_k={TOP_K}")
    print("  SPACE = pause/resume, N = step frame, Q = quit early\n")
 
    start_time = time.time()
 
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break  # end of video
 
            total_frames += 1
            _, faces = detector.detect(frame)
            best = pick_largest(faces)
 
            if best is not None:
                detected_frames += 1
                x, y, w, h = best[0:4].astype(int)
                confidence = best[14]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cx, cy = x + w // 2, y + h // 2
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"{confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # draw the 5 facial landmarks
                landmarks = best[4:14].astype(int).reshape(-1, 2)
                for (lx, ly) in landmarks:
                    cv2.circle(frame, (lx, ly), 2, (0, 255, 255), -1)
            else:
                cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
 
            ratio = detected_frames / total_frames if total_frames else 0
            cv2.putText(frame, f"Frame {total_frames}  detected {detected_frames} ({ratio:.1%})",
                        (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
 
            if writer is not None:
                writer.write(frame)
 
            current_frame = frame
 
        cv2.imshow("Detection benchmark (YuNet) - Q to quit, SPACE to pause", current_frame)
 
        key = cv2.waitKey(0 if paused else frame_delay_ms) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n') and paused:
            paused = False
            cv2.waitKey(1)
            paused = True
 
    elapsed = time.time() - start_time
 
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
 
    print("\n--- Summary ---")
    print(f"Total frames processed: {total_frames}")
    print(f"Frames with a detection: {detected_frames}")
    if total_frames:
        print(f"Detection rate: {detected_frames / total_frames:.1%}")
    print(f"Processing time: {elapsed:.1f}s")
    if SAVE_OUTPUT_PATH:
        print(f"Annotated video saved to: {SAVE_OUTPUT_PATH}")
 
 
if __name__ == "__main__":
    main()