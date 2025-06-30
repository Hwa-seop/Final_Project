import cv2
import numpy as np
import torch
import time
from scipy.spatial import distance as dist
from collections import OrderedDict

# CentroidTracker 정의
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


# 모델 로드
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') # 커스텀 모델
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model.conf = 0.4
model.eval()

tracker = CentroidTracker(max_disappeared=40)
cap = cv2.VideoCapture(0)

roi_points = []
zone_locked = False
zone_poly = None

def mouse_callback(event, x, y, flags, param):
    global roi_points, zone_locked, zone_poly
    if not zone_locked:
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and roi_points:
            roi_points.pop()

def roi_setup():
    global zone_locked, zone_poly

    print("[INFO] Draw ROI polygon with mouse. Enter/Space to confirm, 'q' to quit.")
    cv2.namedWindow("ROI Setup", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Setup", mouse_callback)

    while not zone_locked:
        ret, frame = cap.read()
        if not ret:
            break
        output = frame.copy()

        for pt in roi_points:
            cv2.circle(output, pt, 5, (255, 0, 0), -1)
        if len(roi_points) > 1:
            cv2.polylines(output, [np.array(roi_points, dtype=np.int32).reshape((-1,1,2))],
                          isClosed=False, color=(0, 255, 255), thickness=2)

        cv2.putText(output, "Enter/Space: confirm ROI, q: quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow("ROI Setup", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            exit()
        elif (key == 13 or key == 32) and len(roi_points) >= 3:
            zone_poly = np.array(roi_points, dtype=np.int32).reshape((-1,1,2))
            zone_locked = True

    cv2.destroyWindow("ROI Setup")

def main_loop():
    frame_count = 0
    last_boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 15 == 0:
            results = model(frame)
            dets = results.xyxy[0].cpu().numpy()
            last_boxes = []
            for det in dets:
                x1, y1, x2, y2, conf, cls = det[:6]
                cls = int(cls)
                if conf < 0.5 or cls != 0:  # person만 추적
                    continue
                last_boxes.append((int(x1), int(y1), int(x2), int(y2), cls))

        objects = tracker.update([(x1, y1, x2, y2) for (x1, y1, x2, y2, cls) in last_boxes])

        for (object_id, centroid) in objects.items():
            for (x1, y1, x2, y2, cls) in last_boxes:
                cX = int((x1 + x2) / 2)
                cY = int((y1 + y2) / 2)
                if abs(centroid[0] - cX) < 10 and abs(centroid[1] - cY) < 10:
                    inside = cv2.pointPolygonTest(zone_poly, (cX, cY), False) if zone_locked else -1
                    color = (0, 0, 255) if inside >= 0 else (0, 255, 0)
                    label = f"ID:{object_id}" + (" DANGER" if inside >= 0 else "")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.circle(frame, (cX, cY), 4, (255, 0, 0), -1)
                    break

        if zone_locked and zone_poly is not None:
            cv2.polylines(frame, [zone_poly], isClosed=True, color=(0,0,255), thickness=3)

        cv2.imshow("Centroid Helmet Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    roi_setup()
    main_loop()
