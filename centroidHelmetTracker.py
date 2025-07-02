# pip install torch torchvision opencv-python numpy scipy

import cv2
import torch
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
from typing import Any

# IoU 계산 함수
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])  # 좌측 상단 x 좌표 중 큰 값
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])  # 우측 하단 x 좌표 중 작은 값
    yB = min(boxA[3], boxB[3])  # 우측 하단 y 좌표 중 작은 값
    interArea = max(0, xB - xA) * max(0, yB - yA)  # 겹치는 영역의 넓이
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# 객체 중심 좌표 기반 단순 추적기
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = dist.cdist(np.array(object_centroids), input_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])) - used_rows
            unused_cols = set(range(0, D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

# YOLOv5 사용자 모델 로드
model: Any = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.conf = 0.4
model.eval()

cap = cv2.VideoCapture(0)
tracker = CentroidTracker(max_disappeared=40)
frame_count = 0
person_boxes = []
helmet_boxes = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 5 == 0:
        results = model(frame)
        dets = results.xyxy[0].cpu().numpy()

        person_boxes = []
        helmet_boxes = []

        for det in dets:
            x1, y1, x2, y2, conf, cls = det[:6]
            cls = int(cls)
            if conf < 0.5 or cls not in [0, 1, 2]:
                continue
            w, h = x2 - x1, y2 - y1
            if w < 30 or h < 30:
                continue

            if cls == 0:
                person_boxes.append((int(x1), int(y1), int(x2), int(y2)))
            elif cls == 1:
                helmet_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    objects = tracker.update(person_boxes)

    for object_id, centroid in objects.items():
        matched_person_box = None
        for (x1, y1, x2, y2) in person_boxes:
            cX = (x1 + x2) // 2
            cY = (y1 + y2) // 2
            if abs(centroid[0] - cX) < 10 and abs(centroid[1] - cY) < 10:
                matched_person_box = (x1, y1, x2, y2)
                break

        if matched_person_box is None:
            continue

        x1, y1, x2, y2 = matched_person_box
        person_box = (x1, y1, x2, y2)

        helmet_worn = False
        for helmet_box in helmet_boxes:
            iou = compute_iou(person_box, helmet_box)
            if iou > 0.1:
                helmet_worn = True
                break

        color = (0, 255, 0) if helmet_worn else (0, 0, 255)
        label = "Helmet" if helmet_worn else "No Helmet"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ID:{object_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Helmet Detection Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
