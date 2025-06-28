# pip install onnxruntime opencv-python numpy scipy

import cv2  # 영상 처리 라이브러리
import onnxruntime as ort  # ONNX 모델 추론용 라이브러리
import numpy as np  # 수치 계산
from scipy.spatial import distance as dist  # 유클리드 거리 계산
from collections import OrderedDict  # 순서 있는 딕셔너리 (ID 추적용)

# ----------------------------
# CentroidTracker 클래스 정의: 객체 중심 좌표로 ID 추적 관리
# ----------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        # 객체 ID 초기화 및 추적 정보 저장 딕셔너리 정의
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        # 새 객체 등록 (ID 할당 및 중심점 저장)
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        # 오래 사라진 객체 제거
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        # 감지된 박스가 없으면 모든 객체에 대해 사라짐 횟수 증가
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # 감지된 박스로부터 중심점 계산
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        # 추적 중인 객체가 없으면 모두 새로 등록
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # 기존 객체와 새로운 중심점 사이 거리 계산
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # 가장 가까운 거리 순으로 매칭
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

            # 매칭되지 않은 기존 객체 → 사라짐 증가
            unused_rows = set(range(0, D.shape[0])) - used_rows
            unused_cols = set(range(0, D.shape[1])) - used_cols
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # 매칭되지 않은 새로운 객체 → 새 등록
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

# ----------------------------
# ONNX 추론 보조 함수 정의
# ----------------------------
def preprocess(frame):
    # 프레임을 YOLO 입력 형태로 전처리 (RGB, 정규화, 채널 변경)
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def postprocess(outputs, frame_shape, conf_thres=0.5):
    # ONNX 출력값 후처리 → 사람 클래스(0)만 추출
    pred = outputs[0]
    if pred.ndim == 3:
        pred = pred[0]
    boxes = []
    for det in pred:
        x1, y1, x2, y2, conf, cls = det[:6].tolist()
        if conf < conf_thres or int(cls) != 0:
            continue
        w, h = x2 - x1, y2 - y1
        if w < 30 or h < 30:  # 너무 작은 박스 무시
            continue
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes

# ----------------------------
# 메인 실행부: 모델 로딩, 추론, 시각화
# ----------------------------
session = ort.InferenceSession("yolov5n.onnx")  # ONNX 모델 로딩 (사람 탐지용)
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)  # 웹캠 연결
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)  # 해상도 낮춤
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

tracker = CentroidTracker(max_disappeared=40)  # 추적기 초기화
frame_count = 0
last_boxes = []  # 최근 감지 박스 유지

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 3프레임마다 추론 실행
    if frame_count % 3 == 0:
        img_input = preprocess(frame)
        outputs = session.run(None, {input_name: img_input})
        last_boxes = postprocess(outputs, frame.shape)

    # 추적 업데이트
    objects = tracker.update(last_boxes)

    # 추적 ID 시각화
    for (object_id, centroid) in objects.items():
        for (x1, y1, x2, y2) in last_boxes:
            cX = int((x1 + x2) / 2)
            cY = int((y1 + y2) / 2)
            if abs(centroid[0] - cX) < 10 and abs(centroid[1] - cY) < 10:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {object_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                break

    cv2.imshow("YOLOv5 ONNX + Centroid", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
