from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import joblib
from collections import deque
import time

app = Flask(__name__)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False
screen_points = np.array([
    [100, 100],
    [screen_w - 100, 100],
    [screen_w // 2, screen_h // 2],
    [100, screen_h - 100],
    [screen_w - 100, screen_h - 100]
], dtype=np.float32)

camera_points = []
calibrated = False
calib_step = 0
M = None

smooth_buffer = deque(maxlen=30)

blink_count = 0
eye_closed = False
closed_frames = 0
fatigue_message = ""
BLINK_THRESHOLD = 20
CLOSED_FRAMES_LIMIT = 20

fps = camera.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
blink_history = deque(maxlen=int(fps * 60))



# KNN model
try:
    knn = joblib.load("knn_gaze_model.pkl")
    scaler = joblib.load("scaler_gaze.pkl")
except Exception as e:
    knn = None
    scaler = None
    print("Atenție: model KNN/scaler neîncărcate:", e)


def get_iris_position(face_landmarks, image_w, image_h):
    """poz irisului stg fol pt calibrare"""
    left_eye_outer = face_landmarks.landmark[33]
    left_eye_inner = face_landmarks.landmark[133]
    left_iris = face_landmarks.landmark[468]

    x_outer = int(left_eye_outer.x * image_w)
    x_inner = int(left_eye_inner.x * image_w)
    x_iris = int(left_iris.x * image_w)
    y_iris = int(left_iris.y * image_h)
    # coordonate reale inn pixeli din img

    gaze_ratio = (x_iris - x_outer) / max((x_inner - x_outer), 1e-6)
    dot_x = int(gaze_ratio * image_w)
    dot_y = y_iris
    return dot_x, dot_y


def get_iris_relative(landmarks):
    """coord relative pt KNN"""
    # stâng
    left_outer = landmarks[33]
    left_inner = landmarks[133]
    left_top = landmarks[159]
    left_bottom = landmarks[145]
    left_center_x = (left_outer.x + left_inner.x) / 2
    left_center_y = (left_top.y + left_bottom.y) / 2
    left_w = (left_inner.x - left_outer.x) if (left_inner.x - left_outer.x) != 0 else 1e-6
    left_h = (left_bottom.y - left_top.y) if (left_bottom.y - left_top.y) != 0 else 1e-6
    left_iris = landmarks[468]
    lx_rel = (left_iris.x - left_center_x) / left_w
    ly_rel = (left_iris.y - left_center_y) / left_h

    # drept
    right_outer = landmarks[362]
    right_inner = landmarks[263]
    right_top = landmarks[386]
    right_bottom = landmarks[374]
    right_center_x = (right_outer.x + right_inner.x) / 2
    right_center_y = (right_top.y + right_bottom.y) / 2
    right_w = (right_inner.x - right_outer.x) if (right_inner.x - right_outer.x) != 0 else 1e-6
    right_h = (right_bottom.y - right_top.y) if (right_bottom.y - right_top.y) != 0 else 1e-6
    right_iris = landmarks[473]
    rx_rel = (right_iris.x - right_center_x) / right_w
    ry_rel = (right_iris.y - right_center_y) / right_h

    return lx_rel, ly_rel, rx_rel, ry_rel


def draw_eye_overlay(frame, landmarks):
    h, w, _ = frame.shape

    left_pts_idx = [33, 133, 159, 145]
    right_pts_idx = [362, 263, 386, 374]

    left_pts = []
    for idx in left_pts_idx:
        lm = landmarks.landmark[idx]
        left_pts.append((int(lm.x * w), int(lm.y * h)))
    right_pts = []
    for idx in right_pts_idx:
        lm = landmarks.landmark[idx]
        right_pts.append((int(lm.x * w), int(lm.y * h)))

    cv2.polylines(frame, [np.array(left_pts)], True, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.polylines(frame, [np.array(right_pts)], True, (0, 0, 255), 1, cv2.LINE_AA)

    try:
        left_iris = landmarks.landmark[468]
        right_iris = landmarks.landmark[473]
        lx_pix = int(left_iris.x * w)
        ly_pix = int(left_iris.y * h)
        rx_pix = int(right_iris.x * w)
        ry_pix = int(right_iris.y * h)
        cv2.circle(frame, (lx_pix, ly_pix), 3, (0, 255, 0), -1)
        cv2.circle(frame, (rx_pix, ry_pix), 3, (0, 255, 0), -1)
    except Exception:
        pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/affine')
def affine():
    return render_template('affine.html')


@app.route('/knn')
def knn_page():
    return render_template('knn.html')


def gen_frames_affine():
    global calibrated, calib_step, camera_points, M
    global blink_count, eye_closed, closed_frames, fatigue_message

    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        image_h, image_w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        fatigue_message = ""

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            draw_eye_overlay(frame, face_landmarks)

            top = face_landmarks.landmark[159]
            bottom = face_landmarks.landmark[145]
            top_y = int(top.y * image_h)
            bottom_y = int(bottom.y * image_h)
            eye_distance = abs(top_y - bottom_y)

            if eye_distance < 5:
                if not eye_closed:
                    blink_count += 1
                    eye_closed = True
                    blink_history.append(1)
                closed_frames += 1
            else:
                eye_closed = False
                closed_frames = 0
                blink_history.append(0)

            if closed_frames >= CLOSED_FRAMES_LIMIT:
                fatigue_message = "OBOSIT: ochii inchisi prea mult!"
            elif sum(blink_history) > BLINK_THRESHOLD:
                fatigue_message = "OBOSIT: prea multe clipiri!"

            cv2.putText(frame, f"Clipiri: {blink_count}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            if fatigue_message:
                cv2.putText(frame, fatigue_message, (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            dot_x, dot_y = get_iris_position(face_landmarks, image_w, image_h)

            if not calibrated:
                # pct rosu pt mapare
                tx, ty = screen_points[calib_step]
                draw_x = int(tx * image_w / screen_w)
                draw_y = int(ty * image_h / screen_h)
                cv2.circle(frame, (draw_x, draw_y), 15, (0, 0, 255), -1)
            else:
                # aplicare estimarea afina
                p = np.array([dot_x, dot_y, 1.0])
                mapped = M.dot(p)
                screen_x = int(mapped[0])
                screen_y = int(mapped[1])

                smooth_buffer.append((screen_x, screen_y))
                avg_x = int(np.mean([p[0] for p in smooth_buffer]))
                avg_y = int(np.mean([p[1] for p in smooth_buffer]))

                pyautogui.moveTo(avg_x, avg_y)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def gen_frames_knn():
    global blink_count, eye_closed, closed_frames, fatigue_message
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        image_h, image_w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        fatigue_message = ""

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # desen ochi + iris
            draw_eye_overlay(frame, landmarks)

            top = landmarks.landmark[159]
            bottom = landmarks.landmark[145]
            top_y = int(top.y * image_h)
            bottom_y = int(bottom.y * image_h)
            eye_distance = abs(top_y - bottom_y)

            if eye_distance < 5:
                if not eye_closed:
                    blink_count += 1
                    eye_closed = True
                    blink_history.append(1)
                closed_frames += 1
            else:
                eye_closed = False
                closed_frames = 0
                blink_history.append(0)

            if closed_frames >= CLOSED_FRAMES_LIMIT:
                fatigue_message = "OBOSIT: ochii inchisi prea mult!"
            elif sum(blink_history) > BLINK_THRESHOLD:
                fatigue_message = "OBOSIT: prea multe clipiri!"

            cv2.putText(frame, f"Clipiri: {blink_count}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            if fatigue_message:
                cv2.putText(frame, fatigue_message, (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # predictie KNN daca modelul e incarcat
            if knn is not None and scaler is not None:
                lx, ly, rx, ry = get_iris_relative(landmarks.landmark)
                X = scaler.transform([[lx, ly, rx, ry]])
                pred = knn.predict(X)[0]
                cv2.putText(frame, f"Privire: {pred}", (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "KNN model not loaded", (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed_affine')
def video_feed_affine():
    return Response(gen_frames_affine(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_knn')
def video_feed_knn():
    return Response(gen_frames_knn(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/keypress/<key>', methods=['POST'])
def keypress(key):
    global calibrated, calib_step, camera_points, M
    if key == "space" and not calibrated:
        success, frame = camera.read()
        if success:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                dot_x, dot_y = get_iris_position(face_landmarks, w, h)
                camera_points.append([dot_x, dot_y])
                if calib_step < len(screen_points) - 1:
                    calib_step += 1
                else:
                    calib_step = len(camera_points)

                #dupa ce am facut toate salavarile de puncte, calculam M
                if len(camera_points) == len(screen_points):
                    cam_pts = np.array(camera_points, dtype=np.float32)
                    M, _ = cv2.estimateAffine2D(cam_pts, screen_points)
                    calibrated = True
                    print("Calibrare completă. Matricea M:\n", M)
    elif key == "c" and calibrated:
        if smooth_buffer:
            avg_x = int(np.mean([p[0] for p in smooth_buffer]))
            avg_y = int(np.mean([p[1] for p in smooth_buffer]))
            pyautogui.click(avg_x, avg_y)
    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
