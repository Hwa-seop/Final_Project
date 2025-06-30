import cv2
import torch
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

# ------------------ Centroid Tracker ------------------ #
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

# ------------------ ROI Mouse Callback ------------------ #
roi_points = []
zone_locked = False
zone_poly = None

def mouse_callback(event, x, y, flags, param):
    global roi_points, zone_locked
    if not zone_locked:
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and roi_points:
            roi_points.pop()

# ------------------ Model & Camera Init ------------------ #
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # best.pt는 사용자 학습 모델
model.conf = 0.4
model.eval()

cap = cv2.VideoCapture(0)
tracker = CentroidTracker(max_disappeared=40)
frame_count = 0
last_boxes = []

# ------------------ ROI 설정 루프 ------------------ #
cv2.namedWindow("Draw ROI")
cv2.setMouseCallback("Draw ROI", mouse_callback)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    temp = frame.copy()
    for pt in roi_points:
        cv2.circle(temp, pt, 5, (255, 0, 0), -1)
    if len(roi_points) > 1:
        cv2.polylines(temp, [np.array(roi_points, np.int32).reshape((-1,1,2))], False, (0,255,255), 2)

    cv2.putText(temp, "Left click: add, Right click: undo, Enter: confirm", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.imshow("Draw ROI", temp)
    key = cv2.waitKey(1) & 0xFF
    if key in [13, 32] and len(roi_points) >= 3:
        zone_poly = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))
        zone_locked = True
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Draw ROI")

# ------------------ 메인 루프 ------------------ #
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    alert_active = False

    if frame_count % 15 == 0:
        results = model(frame)
        dets = results.xyxy[0].cpu().numpy()
        last_boxes = []
        for det in dets:
            x1, y1, x2, y2, conf, cls = det[:6]
            cls = int(cls)
            if conf < 0.5 or cls not in [0, 1, 2]:
                continue
            w, h = x2 - x1, y2 - y1
            if w < 30 or h < 30:
                continue
            last_boxes.append((int(x1), int(y1), int(x2), int(y2), cls))

    rects = [(x1, y1, x2, y2) for (x1, y1, x2, y2, cls) in last_boxes]
    objects = tracker.update(rects)

    for (object_id, centroid) in objects.items():
        for (x1, y1, x2, y2, cls) in last_boxes:
            cX = int((x1 + x2) / 2)
            cY = int((y1 + y2) / 2)
            if abs(centroid[0] - cX) < 10 and abs(centroid[1] - cY) < 10:
                label = model.names[cls]
                color = (0, 255, 0) if label == 'helmet' else (0, 0, 255) if label == 'no-helmet' else (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ID:{object_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if zone_locked and cv2.pointPolygonTest(zone_poly, (centroid[0], centroid[1]), False) >= 0:
                    alert_active = True
                break

    if zone_locked:
        poly_color = (0, 0, 255) if alert_active else (0, 255, 0)
        cv2.polylines(frame, [zone_poly], True, poly_color, 3)
        if alert_active:
            cv2.putText(frame, "DANGER!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Helmet Tracker + ROI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
