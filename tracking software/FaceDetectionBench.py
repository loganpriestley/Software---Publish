
import cv2
import time


VIDEO_PATH = r"C:\Users\33log\OneDrive\Desktop\TrackingCamera\Software\SweepTestFootage.mp4"


SCALE_FACTOR = 0.75
process_fac = 1.05
neighbors = 3
MIN_FACE_SIZE = (60, 60)


SAVE_OUTPUT_PATH = "annotated_output.mp4" 



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_face(frame):
    
    small = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=process_fac,
        minNeighbors=neighbors,
        minSize=(int(MIN_FACE_SIZE[0] * SCALE_FACTOR), int(MIN_FACE_SIZE[1] * SCALE_FACTOR))
    )

    if len(faces) == 0:
        return None

    largest = faces[0]
    for face in faces:
        if largest[2] * largest[3] < face[2] * face[3]:
            largest = face

    x = int(largest[0] / SCALE_FACTOR)
    y = int(largest[1] / SCALE_FACTOR)
    w = int(largest[2] / SCALE_FACTOR)
    h = int(largest[3] / SCALE_FACTOR)

    center_x = x + w // 2
    center_y = y + h // 2

    return center_x, center_y, w, h


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Could not open video file: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay_ms = int(1000 / fps)

    writer = None
    if SAVE_OUTPUT_PATH:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(SAVE_OUTPUT_PATH, fourcc, fps, (w, h))

    total_frames = 0
    detected_frames = 0
    paused = False

    print("Running benchmark...")
    print(f"  video: {VIDEO_PATH}")
    print(f"  process_fac={process_fac}  neighbors={neighbors}  SCALE_FACTOR={SCALE_FACTOR}  MIN_FACE_SIZE={MIN_FACE_SIZE}")
    print("  SPACE = pause/resume, N = step frame, Q = quit early\n")

    start_time = time.time()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break  # end of video

            total_frames += 1
            detected = detect_face(frame)

            if detected is not None:
                detected_frames += 1
                cx, cy, w, h = detected
                top_left = (cx - w // 2, cy - h // 2)
                bottom_right = (cx + w // 2, cy + h // 2)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            else:
                cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # running stats overlay
            ratio = detected_frames / total_frames if total_frames else 0
            cv2.putText(frame, f"Frame {total_frames}  detected {detected_frames} ({ratio:.1%})",
                        (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if writer is not None:
                writer.write(frame)

            current_frame = frame

        cv2.imshow("Detection benchmark - Q to quit, SPACE to pause", current_frame)

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