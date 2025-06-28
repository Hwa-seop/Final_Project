# git clone https://github.com/ultralytics/yolov5
# cd yolov5
# pip install -r requirements.txt
# python export.py --weights yolov5n.pt --include onnx

# sudo apt install libopenblas-dev libopenmpi-dev
# pip install onnxruntime opencv-python numpy scipy

import cv2
import onnxruntime as ort
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

# Centroid 클래스 설정
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

# onnx 함수선언
def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def postprocess(outputs, frame_shape, conf_thres=0.4):
    pred = outputs[0]
    boxes = []
    for det in pred:
        x1, y1, x2, y2, conf, cls = det[:6]
        if conf < conf_thres or int(cls) != 0:  # 사람만
            continue
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes

# main
session = ort.InferenceSession("yolov5n.onnx")
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

tracker = CentroidTracker(max_disappeared=40)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_input = preprocess(frame)
    outputs = session.run(None, {input_name: img_input})
    boxes = postprocess(outputs, frame.shape)

    objects = tracker.update(boxes)

    for (object_id, centroid) in objects.items():
        for (x1, y1, x2, y2) in boxes:
            cX = int((x1 + x2) / 2)
            cY = int((y1 + y2) / 2)
            if abs(centroid[0] - cX) < 10 and abs(centroid[1] - cY) < 10:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {object_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                break

    cv2.imshow("jetson yolo5 onnx centroid", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
