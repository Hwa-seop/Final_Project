import cv2
import numpy as np
import torch
from scipy.spatial import distance as dist
from collections import OrderedDict

# ---------------------------
# Centroid Tracker Class
# ---------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=40):
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

# ---------------------------
# Load Two YOLO Models
# ---------------------------
person_model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
person_model.conf = 0.4
person_model.eval()

helmet_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
helmet_model.conf = 0.4
helmet_model.eval()

# ---------------------------
# Initialize Camera & Tracker
# ---------------------------
cap = cv2.VideoCapture(0)
tracker = CentroidTracker(max_disappeared=40)
frame_count = 0
last_boxes = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    person_boxes, helmet_boxes = [], []

    if frame_count % 15 == 0:
        # Person detection
        person_results = person_model(frame)
        person_dets = person_results.xyxy[0].cpu().numpy()
        for det in person_dets:
            x1, y1, x2, y2, conf, cls = det[:6]
            if int(cls) == 0 and conf >= 0.5:  # class 0: person
                person_boxes.append((int(x1), int(y1), int(x2), int(y2), int(cls)))

        # Helmet detection
        helmet_results = helmet_model(frame)
        helmet_dets = helmet_results.xyxy[0].cpu().numpy()
        for det in helmet_dets:
            x1, y1, x2, y2, conf, cls = det[:6]
            if conf >= 0.5 and int(cls) in [1, 2]:  # helmet, no-helmet
                helmet_boxes.append((int(x1), int(y1), int(x2), int(y2), int(cls)))

        last_boxes = person_boxes + helmet_boxes

    tracked_objects = tracker.update([(x1, y1, x2, y2) for (x1, y1, x2, y2, cls) in last_boxes])

    for (object_id, centroid) in tracked_objects.items():
        for (x1, y1, x2, y2, cls) in last_boxes:
            cX = int((x1 + x2) / 2)
            cY = int((y1 + y2) / 2)
            if abs(centroid[0] - cX) < 10 and abs(centroid[1] - cY) < 10:
                if cls == 0:
                    label = "person"
                    color = (0, 255, 0)
                elif cls == 1:
                    label = "helmet"
                    color = (255, 255, 0)
                else:
                    label = "no-helmet"
                    color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ID:{object_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                break

    cv2.imshow("Dual YOLO Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
