#import everything
import cv2
import serial
import time

#configuration constants

SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
CAMERA_INDEX = 1

#control loop variables
pan_output=0
tilt_output=0
#pan factors
pan_kp = .01
pan_ki = .0001
pan_kd = .004

#tilt factors
tilt_kp = .01
tilt_ki = .0001
tilt_kd = .004


#error trackers and integrals
last_pan_error = 0
pan_integral = 0

last_tilt_error = 0
tilt_integral = 0

#lost searching variable initilizations
pan_direction = 0
tilt_direction = 0
missed_frames = 0
found_frames = 0
roaming = False
is_lost = False
MIN_ANGLE_SEARCH=40
MAX_ANGLE_SEARCH=130

#number of consecutive frames required to change teh state to prevent single missed frames from causing errors
MISS_FRAMES_THRESHOLD = 8      
CONFIRM_FRAMES_THRESHOLD = 3   


#servo stepping constants
MIN_STEP=1
MAX_STEP=10
MIN_ANGLE=0
MAX_ANGLE=170

#detection constants:
MODEL_PATH = "face_detection_yunet_2026may.onnx"   
SCORE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.3
TOP_K = 5000

#Serial/servo variable intializations
pan =90
tilt=90
send_interval = .08

last_pid_time = 0

#set up serial connection and handle any potential error by allowing user to choose different ports
while True:
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        break
    except serial.SerialException as e:
        print(f"Could not open serial port {SERIAL_PORT}: {e}")
        SERIAL_PORT = input("Enter new port (e.g. COM4), or Q to cancel")
        if SERIAL_PORT.strip().upper() == 'Q':
            raise SystemExit(1)


#attempt to open camera and allow user to try indexes til they find the index
cap = cv2.VideoCapture(CAMERA_INDEX)
while not cap.isOpened():
    print(f"Could not open camera at index {CAMERA_INDEX}")
    user_input = input("Enter new index, or Q to quit: ")
    if user_input.strip().upper() == 'Q':
        raise SystemExit(1)
    CAMERA_INDEX = int(user_input)
    cap = cv2.VideoCapture(CAMERA_INDEX)


#YuNet needs to know the frame size up front
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

face_detector = cv2.FaceDetectorYN.create(
    MODEL_PATH, "", (cam_width, cam_height),
    SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
)


def send_angles(p, t):
    """
    Sends the calculated angles to the esp32

    Args:
        p: angle for the pan servo
        t: angle for the tilt servo
    """
    command = f"{p},{t}\n"
    ser.write(command.encode('utf-8'))


def detect_face(frame):
    """
    Takes the raw frame and finds the largest face using YuNet.

    Args:
        frame: the frame captured by the camera

    Returns:
        center_x: the x coordinate of the faces center
        center_y: the y coordinate of the faces center
        w: the width of the faces bounding box
        h: the height of the faces boundign box
    """

    
    _, faces = face_detector.detect(frame)

    if faces is None or len(faces) == 0:
        return None

    #choose the largest face
    largest = faces[0]
    for face in faces:
        if largest[2] * largest[3] < face[2] * face[3]:
            largest = face

    x, y, w, h = largest[0:4].astype(int)

    #find center
    center_x = x + w // 2
    center_y = y + h // 2

    return center_x, center_y, w, h

def search_when_lost(pan, tilt, pan_direction, tilt_direction,roaming, pan_step=.6, tilt_step=10, search_step=1):
    """
    When a face is not detected it continues oving in the direction 
    it last was and when it hits the bounds of the servos rotation it starts
    to roam around

    Args:
        pan, tilt: current servo angles
        pan_direction, tilt_direction: +1, -1, or 0 the last known direction of travel for each axis
        roaming: bool, whether we've already hit a limit and switched into full-sweep search mode
        pan_step: step in pan direction when roaming
        tilt_step: step in tilt direction while roaming. only used after a full pan is complete
        search_step: step when lost but not roaming

    Returns:
        pan, tilt: updated angles
        pan_direction, tilt_direction: updated directions (flipped at the limits while roaming, so it reverses instead of sticking)
        roaming: updated roaming flag
    """
    if not roaming:
        if pan_direction != 0:
            pan -= pan_direction * search_step
        if tilt_direction != 0:
            tilt -= tilt_direction * search_step

        if pan <= MIN_ANGLE or pan >= MAX_ANGLE or tilt <= MIN_ANGLE or tilt >= MAX_ANGLE:
            roaming = True

            if pan_direction >=0:
                pan_direction = 1
            else:
                pan_direction=-1
            if tilt_direction >=0:
                tilt_direction = 1
            else:
                tilt_direction = -1
            
    else:
        pan += pan_direction * pan_step

        if pan <= MIN_ANGLE or pan >= MAX_ANGLE:
            pan = max(MIN_ANGLE, min(MAX_ANGLE, pan))
            pan_direction *= -1

            tilt += tilt_direction * tilt_step
            if tilt <= MIN_ANGLE_SEARCH or tilt >= MAX_ANGLE_SEARCH:
                tilt = max(MIN_ANGLE_SEARCH, min(MAX_ANGLE_SEARCH, tilt))
                tilt_direction *= -1

    pan = max(MIN_ANGLE, min(MAX_ANGLE, pan))
    tilt = max(MIN_ANGLE_SEARCH, min(MAX_ANGLE_SEARCH, tilt))

    return pan, tilt, pan_direction, tilt_direction, roaming

def compute_pid(error, last_error, integral, kp, ki, kd, dt, max_integral_output=5):
    """
    Computes the PID step for one axis

    Args:
        error: the current distance in pixels between the center of the frame and the center of the face bounding box
        last_error: error from the previous call to be used for the derivative term
        integral: the accumulated integral term
        kp,ki,kd: PID gains
        dt: the time since the last call to be used for derivative term
        max_integral_output: the maximum the integral term alone is allowed to
            contribute to output, in the same units as output (degrees per
            step). Defaults to half of MAX_STEP, so integral windup can never
            account for more than half a normal step regardless of how long
            error has been building or what ki happens to be tuned to.

    Returns:
        output: the calcualted adjustment for the axis in question
        integral: updated integral to be passed back in
        error: the error to be passed back in as last_error in the next call
    """
    if max_integral_output is None:
        max_integral_output = MAX_STEP * 0.5

    p_term = error*kp
    integral+= error*dt

    #clamp based on the integral's actual contribution to output 
    if ki != 0:
        integral_limit = max_integral_output / ki
        integral = max(-integral_limit, min(integral_limit, integral))

    i_term = integral *ki

    if dt>0:
        d_term = kd * (error-last_error)/dt
    else:
        d_term = 0

    output = p_term + i_term + d_term

    return output, integral, error


send_angles(pan, tilt)

frame_h, frame_w = None, None
print("Face tracking started.")


while(True):

    #attempt to grab frame
    ret, frame = cap.read()

    #skip if unsuccessful
    if not ret:
        continue

    #set size of the frame
    if frame_h is None:
        frame_h, frame_w = frame.shape[:2]

    #set the time of the loop so all time checks are consistent
    now = time.monotonic()

    #run face detection
    detected = detect_face(frame)

    #update the debounce counters every frame based on this frames raw result
    if detected is not None:
        found_frames += 1
        missed_frames = 0
    else:
        missed_frames += 1
        found_frames = 0

    #only flip the trusted is_lost state once a counter crosses its threshold
    #this is what keeps a single dropped frame or single false-positive frame
    #from immediately changing behavior
    if not is_lost and missed_frames >= MISS_FRAMES_THRESHOLD:
        is_lost = True
    elif is_lost and found_frames >= CONFIRM_FRAMES_THRESHOLD:
        is_lost = False
        roaming = False

    if detected is not None and not is_lost:
        #draw rectangle based on the returned values from the detection
        #top left corner smaller x smaller y
        start_point = (detected[0] - detected[2] // 2, detected[1] - detected[3] // 2)

        #bottom right corner larger x larger y
        end_point = (detected[0] + detected[2] // 2, detected[1] + detected[3] // 2)

        cv2.rectangle(frame,start_point,end_point,(0,255,0),2)

        #check timing and start the pid process
        if(now-last_pid_time>send_interval):

            #calcualte how far box is from center
            pan_difference = detected[0]-(frame_w/2)

            tilt_difference = detected[1]-(frame_h//2)

            #find difference in time from the last pid computation
            dt = min(now - last_pid_time, 0.2)

            #compute pid for the pan
            pan_output, pan_integral, last_pan_error = compute_pid(pan_difference,last_pan_error,pan_integral, pan_kp,pan_ki,pan_kd,dt)

            #compute pid for the tilt
            tilt_output, tilt_integral, last_tilt_error = compute_pid(tilt_difference,last_tilt_error,tilt_integral, tilt_kp,tilt_ki,tilt_kd,dt)

            #keep step size within bounds of the max step
            pan_output  = max(-MAX_STEP, min(MAX_STEP, pan_output))
            tilt_output = max(-MAX_STEP, min(MAX_STEP, tilt_output))

            #process into an angle to be sent to the esp32
            pan_changed = abs(pan_output) >= MIN_STEP
            tilt_changed = abs(tilt_output) >= MIN_STEP

            if pan_changed or tilt_changed:
                if pan_changed:
                    pan = max(MIN_ANGLE, min(MAX_ANGLE, pan - pan_output))
                if tilt_changed:
                    tilt = max(MIN_ANGLE, min(MAX_ANGLE, tilt - tilt_output))

                send_angles(int(pan), int(tilt))


            last_pid_time = now

            roaming = False

            pan_direction = pan_output
            tilt_direction = tilt_output

    elif is_lost:
        pan,tilt,pan_direction,tilt_direction,roaming = search_when_lost(pan,tilt,pan_direction,tilt_direction,roaming)
        send_angles(int(pan), int(tilt))


    #display useful info
    cv2.putText(frame, f"Pan output:  {pan_output} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Tilt output: {tilt_output} deg", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Face Detected: {bool(detected)}q", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # draw the quit hint on the frame
    cv2.putText(frame, "Q = quit", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.imshow("webcam feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


ser.close()
cap.release()
cv2.destroyAllWindows()