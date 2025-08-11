import cv2
import winsound
import mediapipe as mp
from scipy.spatial import distance as dist
import numpy as np
import time
import pyttsx3
import threading
import csv
from datetime import datetime
import os
import math

# --- Constants ---
EAR_THRESHOLD = 0.50
EAR_CONSECUTIVE_FRAMES = 20
YAWN_THRESHOLD = 25
PHONE_NEAR_FACE_THRESHOLD = 200  # pixels

frame_counter = 0
drowsy = False
yawn_start_time = None
phone_detected = False
alert_reset = False
drowsy_beeping = False

# CSV log file
LOG_FILE = "driver_monitor_log.csv"

# Create log file with headers if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Event", "Details"])

def log_event(event_type, details=""):
    """Log an event with timestamp into CSV."""
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event_type, details])

# Voice engine setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Mediapipe setups
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Eye and mouth landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
UPPER_LIP = [13, 312]
LOWER_LIP = [14, 17]
NOSE_TIP = 1  # MediaPipe face mesh index

def eye_aspect_ratio(landmarks, eye_points):
    p1 = np.array(landmarks[eye_points[1]])
    p2 = np.array(landmarks[eye_points[5]])
    p3 = np.array(landmarks[eye_points[2]])
    p4 = np.array(landmarks[eye_points[4]])
    p5 = np.array(landmarks[eye_points[0]])
    p6 = np.array(landmarks[eye_points[3]])
    A = dist.euclidean(p1, p2)
    B = dist.euclidean(p3, p4)
    C = dist.euclidean(p5, p6)
    return (A + B) / (2.0 * C)

def lip_distance(landmarks):
    top = np.array(landmarks[UPPER_LIP[0]])
    bottom = np.array(landmarks[LOWER_LIP[0]])
    return dist.euclidean(top, bottom)

# Continuous beep thread
def continuous_beep():
    global drowsy_beeping
    while drowsy_beeping:
        winsound.Beep(2500, 500)
        time.sleep(0.1)

cap = cv2.VideoCapture(0)
face_landmarks_saved = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(rgb)
    results_hands = hands.process(rgb)
    h, w, _ = frame.shape

    alert_reset = False

    # --- Palm reset detection ---
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            if (hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and
                hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y and
                hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y and
                hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y):
                alert_reset = True

    if alert_reset:
        frame_counter = 0
        drowsy = False
        phone_detected = False
        yawn_start_time = None
        if drowsy_beeping:
            drowsy_beeping = False
        log_event("RESET", "Driver showed palm - all alerts stopped")
        cv2.putText(frame, "ALL GOOD", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        continue

    # --- Store face landmarks ---
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            face_landmarks_saved = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

    # --- Smart phone use detection ---
    if face_landmarks_saved and results_hands.multi_hand_landmarks:
        nose_point = np.array(face_landmarks_saved[NOSE_TIP])
        for hand_landmarks in results_hands.multi_hand_landmarks:
            index_finger_tip = np.array((int(hand_landmarks.landmark[8].x * w),
                                         int(hand_landmarks.landmark[8].y * h)))
            distance_to_face = dist.euclidean(nose_point, index_finger_tip)

            if distance_to_face < PHONE_NEAR_FACE_THRESHOLD:
                if not phone_detected:
                    engine.say("Do not use phone, take vehicle to side and talk.")
                    engine.runAndWait()
                    log_event("PHONE USE", f"Hand {distance_to_face:.1f}px from face")
                phone_detected = True
                cv2.putText(frame, "PHONE USE DETECTED!", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                break
        else:
            phone_detected = False

    # --- Face processing ---
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

            leftEAR = eye_aspect_ratio(landmarks, LEFT_EYE)
            rightEAR = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (leftEAR + rightEAR) / 2.0
            mouth_open = lip_distance(landmarks)

            # Draw eyes & mouth
            for idx in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, landmarks[idx], 2, (0, 255, 0), -1)
            cv2.circle(frame, landmarks[UPPER_LIP[0]], 2, (255, 0, 0), -1)
            cv2.circle(frame, landmarks[LOWER_LIP[0]], 2, (255, 0, 0), -1)

            # --- Drowsiness detection ---
            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= EAR_CONSECUTIVE_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not drowsy_beeping:
                        drowsy_beeping = True
                        threading.Thread(target=continuous_beep, daemon=True).start()
                        log_event("DROWSINESS", "Eyes closed for extended time")
                    drowsy = True
            else:
                frame_counter = 0
                drowsy = False

            # --- Yawn detection ---
            if mouth_open > YAWN_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = time.time()
                else:
                    elapsed = time.time() - yawn_start_time
                    if elapsed >= 10:
                        for _ in range(3):
                            winsound.Beep(1500, 500)
                            time.sleep(0.5)
                        log_event("YAWN", "Mouth open for 10+ seconds")
                        yawn_start_time = None
                cv2.putText(frame, "YAWN DETECTED", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                yawn_start_time = None

    cv2.imshow("Driver Monitor (Smart + Logging)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
drowsy_beeping = False
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
from scipy.spatial import distance as dist

# -------------------- CONFIG --------------------
DRIVING = True  # Pretend we're always driving
ALERT_COOLDOWN = 3  # seconds between alerts
CALIBRATION_TIME = 2  # seconds for auto face size calibration
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
MOUTH_AR_THRESH = 0.75

# -------------------- INIT --------------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
hands = mp_hands.Hands(max_num_hands=1)
tts = pyttsx3.init()

cap = cv2.VideoCapture(0)

COUNTER = 0
last_alert_time = 0
PHONE_NEAR_FACE_THRESHOLD = None

# -------------------- FUNCTIONS --------------------
def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth_points):
    A = dist.euclidean(mouth_points[2], mouth_points[10])
    C = dist.euclidean(mouth_points[0], mouth_points[6])
    return A / C

def speak_alert(msg):
    global last_alert_time
    if time.time() - last_alert_time > ALERT_COOLDOWN:
        print("[ALERT]", msg)
        tts.say(msg)
        tts.runAndWait()
        last_alert_time = time.time()

def calibrate_face_size(frame):
    global PHONE_NEAR_FACE_THRESHOLD
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in face_landmarks.landmark]
        y_coords = [lm.y * h for lm in face_landmarks.landmark]
        face_width = max(x_coords) - min(x_coords)
        PHONE_NEAR_FACE_THRESHOLD = face_width * 0.5  # threshold set as half face width

def is_phone_near_ear(hand_landmarks, face_landmarks, w, h):
    index_tip = np.array([hand_landmarks.landmark[8].x * w,
                          hand_landmarks.landmark[8].y * h])
    ear_point = np.array([face_landmarks.landmark[234].x * w,
                          face_landmarks.landmark[234].y * h])  # left ear landmark
    distance = np.linalg.norm(index_tip - ear_point)
    return distance < PHONE_NEAR_FACE_THRESHOLD

# -------------------- CALIBRATION --------------------
print("[INFO] Calibrating face size...")
start_time = time.time()
while time.time() - start_time < CALIBRATION_TIME:
    ret, frame = cap.read()
    if not ret:
        break
    calibrate_face_size(frame)
print("[INFO] Calibration complete. Threshold =", PHONE_NEAR_FACE_THRESHOLD)

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    # Face detection
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

        # Eye & mouth aspect ratio
        left_eye = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in [33, 160, 158, 133,]()*_*_]()
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Drawing utils
mp_drawing = mp.solutions.drawing_utils

# Flags and thresholds
DRIVING = True  # Kamchalau driving flag
EYE_AR_THRESH = 0.22
MOUTH_AR_THRESH = 0.6
PHONE_NEAR_FACE_THRESHOLD = None  # Will auto-calibrate
ALERT_COOLDOWN = 2  # seconds

last_alert_time = 0
yawn_detected = False

# --- EAR calculation ---
def euclidean_dist(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    return (euclidean_dist(p2, p6) + euclidean_dist(p3, p5)) / (2.0 * euclidean_dist(p1, p4))

def mouth_aspect_ratio(landmarks, mouth_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in mouth_indices]
    return (euclidean_dist(p2, p6) + euclidean_dist(p3, p5)) / (2.0 * euclidean_dist(p1, p4))

# --- Simple hand near ear detection ---
def is_hand_near_ear(hand_landmarks, face_landmarks):
    # Index finger tip
    index_finger = hand_landmarks[8]
    # Nose tip
    nose = face_landmarks[1]
    dist = euclidean_dist(index_finger, nose)
    return dist < PHONE_NEAR_FACE_THRESHOLD

# --- Alert function ---
def trigger_alert(message):
    global last_alert_time
    if time.time() - last_alert_time > ALERT_COOLDOWN:
        print(f"ALERT: {message}")
        winsound.Beep(1000, 500)
        last_alert_time = time.time()

# --- Auto-calibration ---
def calibrate_phone_threshold(cap):
    global PHONE_NEAR_FACE_THRESHOLD
    print("Calibrating... Please sit normally for 2 seconds.")
    start = time.time()
    distances = []

    while time.time() - start < 2:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
                nose = landmarks[1]
                ear_top = landmarks[234]  # Side of face
                distances.append(euclidean_dist(nose, ear_top))

    PHONE_NEAR_FACE_THRESHOLD = (sum(distances) / len(distances)) * 0.6
    print(f"Calibration complete. Phone detection threshold set to {PHONE_NEAR_FACE_THRESHOLD:.2f}")

# --- Main ---
cap = cv2.VideoCapture(0)
calibrate_phone_threshold(cap)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]

            left_eye = [33, 160, 158, 133, 153, 144]
            right_eye = [263, 387, 385, 362, 380, 373]
            mouth = [61, 81, 311, 308, 402, 14]

            left_ear = eye_aspect_ratio(landmarks, left_eye)
            right_ear = eye_aspect_ratio(landmarks, right_eye)
            mar = mouth_aspect_ratio(landmarks, mouth)

            if DRIVING:
                if left_ear < EYE_AR_THRESH and right_ear < EYE_AR_THRESH:
                    trigger_alert("Drowsiness detected!")
                if mar > MOUTH_AR_THRESH and not yawn_detected:
                    yawn_detected = True
                    trigger_alert("Yawning detected!")
                elif mar <= MOUTH_AR_THRESH:
                    yawn_detected = False

            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            if DRIVING and hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    h_landmarks = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]
                    if is_hand_near_ear(h_landmarks, landmarks):
                        trigger_alert("Phone usage detected!")
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Driver Monitoring (Safe drive)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# -------------------
# Init TTS Engine
# -------------------
engine = pyttsx3.init()
engine.setProperty('rate', 170)  # speed

# -------------------
# Mediapipe Setup
# -------------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# -------------------
# Config
# -------------------
DRIVING = True  # pretend we are driving
PHONE_NEAR_FACE_THRESHOLD = None  # will be set via calibration
CALIBRATION_TIME = 2  # seconds

phone_warning_active = False
last_phone_alert_time = 0
PHONE_ALERT_INTERVAL = 2  # seconds between repeated voice alerts

# -------------------
# Functions
# -------------------
def speak(msg):
    engine.say(msg)
    engine.runAndWait()

def calibrate_threshold(cap):
    print("Calibrating face size... Please look straight at the camera.")
    start_time = time.time()
    face_sizes = []
    while time.time() - start_time < CALIBRATION_TIME:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                xs = [lm.x for lm in face_landmarks.landmark]
                ys = [lm.y for lm in face_landmarks.landmark]
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                face_sizes.append((w + h) / 2)
        cv2.imshow("Calibrating...", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyWindow("Calibrating...")
    if face_sizes:
        avg_face_size = sum(face_sizes) / len(face_sizes)
        return avg_face_size * 0.6  # threshold slightly smaller than face size
    return 0.05  # fallback

def detect_phone_near_face(face_landmarks, hand_landmarks):
    face_x = [lm.x for lm in face_landmarks.landmark]
    face_y = [lm.y for lm in face_landmarks.landmark]
    face_cx = sum(face_x) / len(face_x)
    face_cy = sum(face_y) / len(face_y)

    for lm in hand_landmarks.landmark:
        dist_to_face = ((lm.x - face_cx) ** 2 + (lm.y - face_cy) ** 2) ** 0.5
        if dist_to_face < PHONE_NEAR_FACE_THRESHOLD:
            return True
    return False

def palm_shown(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    middle_tip = hand_landmarks.landmark[12]
    if middle_tip.y < wrist.y:  # palm facing camera
        return True
    return False

# -------------------
# Main
# -------------------
cap = cv2.VideoCapture(0)

# Step 1: Calibrate
PHONE_NEAR_FACE_THRESHOLD = calibrate_threshold(cap)
print(f"Calibrated PHONE_NEAR_FACE_THRESHOLD: {PHONE_NEAR_FACE_THRESHOLD}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            if DRIVING and hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    if detect_phone_near_face(face_landmarks, hand_landmarks):
                        if not phone_warning_active:
                            phone_warning_active = True
                            last_phone_alert_time = 0  # force immediate alert

                        if phone_warning_active and (time.time() - last_phone_alert_time > PHONE_ALERT_INTERVAL):
                            speak("Please put your phone down")
                            last_phone_alert_time = time.time()

                    if phone_warning_active and palm_shown(hand_landmarks):
                        phone_warning_active = False
                        print("Phone alert reset")

    cv2.imshow("Driver Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
