import cv2
import mediapipe as mp
import csv
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

filename = "iris_data_relative.csv"
if not os.path.exists(filename):
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lx_rel", "ly_rel", "rx_rel", "ry_rel", "label"])

print("Apasă 1=stanga, 2=mijloc, 3=dreapta, 4=afara pentru a salva poziția ochilor. ESC pentru exit.")

def get_relative_iris(landmarks):
    # Ochi stg
    left_outer = landmarks[33]
    left_inner = landmarks[133]
    left_top = landmarks[159]
    left_bottom = landmarks[145]
    left_center_x = (left_outer.x + left_inner.x) / 2
    left_center_y = (left_top.y + left_bottom.y) / 2
    left_width = left_inner.x - left_outer.x
    left_height = left_bottom.y - left_top.y
    left_iris = landmarks[468]
    lx_rel = (left_iris.x - left_center_x) / left_width
    ly_rel = (left_iris.y - left_center_y) / left_height

    # Ochi drp
    right_outer = landmarks[362]
    right_inner = landmarks[263]
    right_top = landmarks[386]
    right_bottom = landmarks[374]
    right_center_x = (right_outer.x + right_inner.x) / 2
    right_center_y = (right_top.y + right_bottom.y) / 2
    right_width = right_inner.x - right_outer.x
    right_height = right_bottom.y - right_top.y
    right_iris = landmarks[473]
    rx_rel = (right_iris.x - right_center_x) / right_width
    ry_rel = (right_iris.y - right_center_y) / right_height

    return lx_rel, ly_rel, rx_rel, ry_rel

def draw_iris(frame, landmarks, lx_rel, ly_rel, rx_rel, ry_rel):
    h, w, _ = frame.shape

    left_outer = landmarks[33]
    left_inner = landmarks[133]
    left_top = landmarks[159]
    left_bottom = landmarks[145]
    left_cx = (left_outer.x + left_inner.x) / 2 * w
    left_cy = (left_top.y + left_bottom.y) / 2 * h
    left_w = (left_inner.x - left_outer.x) * w
    left_h = (left_bottom.y - left_top.y) * h
    lx_pix = int(left_cx + lx_rel * left_w)
    ly_pix = int(left_cy + ly_rel * left_h)

    right_outer = landmarks[362]
    right_inner = landmarks[263]
    right_top = landmarks[386]
    right_bottom = landmarks[374]
    right_cx = (right_outer.x + right_inner.x) / 2 * w
    right_cy = (right_top.y + right_bottom.y) / 2 * h
    right_w = (right_inner.x - right_outer.x) * w
    right_h = (right_bottom.y - right_top.y) * h
    rx_pix = int(right_cx + rx_rel * right_w)
    ry_pix = int(right_cy + ry_rel * right_h)

    cv2.circle(frame, (lx_pix, ly_pix), 3, (0, 255, 0), -1)
    cv2.circle(frame, (rx_pix, ry_pix), 3, (255, 0, 0), -1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    cv2.line(frame, (w // 3, 0), (w // 3, h), (0, 255, 255), 2)
    cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (0, 255, 255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        lx, ly, rx, ry = get_relative_iris(landmarks)

        draw_iris(frame, landmarks, lx, ly, rx, ry)

        cv2.putText(frame, "1=Stanga  2=Mijloc  3=Dreapta  4=Afara", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        key = cv2.waitKey(1) & 0xFF
        label = None
        if key == ord("1"):
            label = "stanga"
        elif key == ord("2"):
            label = "mijloc"
        elif key == ord("3"):
            label = "dreapta"
        elif key == ord("4"):
            label = "afara"
        elif key == 27:
            break

        if label:
            with open(filename, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([lx, ly, rx, ry, label])
            print(f"[SALVAT] ({lx:.3f}, {ly:.3f}), ({rx:.3f}, {ry:.3f}) -> {label}")

    cv2.imshow("Colectare date iris relativ la fata", frame)

cap.release()
cv2.destroyAllWindows()
