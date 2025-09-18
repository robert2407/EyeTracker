import cv2
import mediapipe as mp
import joblib
import numpy as np

knn = joblib.load("knn_gaze_model.pkl")
scaler = joblib.load("scaler_gaze.pkl")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def get_iris_relative(landmarks):
    # Ochi stg
    left_outer = landmarks[33]
    left_inner = landmarks[133]
    left_center_x = (left_outer.x + left_inner.x) / 2
    left_center_y = (left_outer.y + left_inner.y) / 2
    left_width = left_inner.x - left_outer.x
    left_height = 0.03

    left_iris = landmarks[468]
    lx_rel = (left_iris.x - left_center_x) / left_width
    ly_rel = (left_iris.y - left_center_y) / left_height

    # Ochi drp
    right_outer = landmarks[362]
    right_inner = landmarks[263]
    right_center_x = (right_outer.x + right_inner.x) / 2
    right_center_y = (right_outer.y + right_inner.y) / 2
    right_width = right_inner.x - right_outer.x
    right_height = 0.03

    right_iris = landmarks[473]
    rx_rel = (right_iris.x - right_center_x) / right_width
    ry_rel = (right_iris.y - right_center_y) / right_height

    return lx_rel, ly_rel, rx_rel, ry_rel

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        lx, ly, rx, ry = get_iris_relative(landmarks)

        X_input = np.array([[lx, ly, rx, ry]])
        X_scaled = scaler.transform(X_input)
        pred = knn.predict(X_scaled)[0]

        cv2.putText(frame, f"Privire: {pred}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Eye Tracker Live", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()