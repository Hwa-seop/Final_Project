import cv2
import numpy as np

# 클래스 이름 로드
def read_class_names(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

# 경로 설정
model_path = "/home/lws/kulws2025/kubig2025/final_project/yolov5/helmet-yolo/yolov5s_helmet3/weights/best.onnx"
class_names_file = "/home/lws/kulws2025/kubig2025/final_project/yolov5/helmet-yolo/yolov5s_helmet3/weights/classes.txt"
class_names = read_class_names(class_names_file)

# 모델 로드
net = cv2.dnn.readNetFromONNX(model_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 웹캠 열기
cap = cv2.VideoCapture(0)
cv2.namedWindow("YOLOv5 Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv5 Detection", 1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # 전처리
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    output = outputs[0]

    persons, helmets = [], []

    for i in range(output.shape[1]):
        data = output[0, i]
        conf = data[4]
        if conf < 0.5:
            continue

        class_scores = data[5:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]
        if conf * score < 0.5:
            continue

        cx, cy, w, h = data[0:4]
        x = int((cx - w / 2) * width)
        y = int((cy - h / 2) * height)
        w = int(w * width)
        h = int(h * height)
        box = [x, y, w, h]

        label = class_names[class_id]
        if label == "person":
            persons.append(box)
        elif label == "helmet":
            helmets.append(box)

    # 시각화
    for px, py, pw, ph in persons:
        person_rect = [px, py, px+pw, py+ph]
        has_helmet = False
        for hx, hy, hw, hh in helmets:
            helmet_rect = [hx, hy, hx+hw, hy+hh]
            # 교차 영역 계산
            ix1, iy1 = max(person_rect[0], helmet_rect[0]), max(person_rect[1], helmet_rect[1])
            ix2, iy2 = min(person_rect[2], helmet_rect[2]), min(person_rect[3], helmet_rect[3])
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            if iw * ih > 0:
                has_helmet = True
                cv2.rectangle(frame, (hx, hy), (hx+hw, hy+hh), (255, 0, 0), 2)

        color = (0, 255, 0) if has_helmet else (0, 0, 255)
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), color, 2)

    cv2.imshow("YOLOv5 Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
