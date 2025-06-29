# pip install torch torchvision opencv-python numpy scipy

import cv2  # OpenCV를 사용한 영상 처리
import torch  # PyTorch로 모델 로딩 및 추론
import numpy as np  # 배열 연산
from scipy.spatial import distance as dist  # 거리 계산
from collections import OrderedDict  # 순서를 유지하는 딕셔너리 구조 사용

# ----------------------------
# CentroidTracker 클래스 정의: 객체 중심 좌표로 ID 추적 관리
# ----------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        # 초기 ID 및 저장 공간 초기화
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        # 새 객체 등록 (ID 부여 및 centroid 저장)
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        # 오래 사라진 객체 제거
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        # 감지된 객체가 없으면 disappear count 증가
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # 입력 박스 중심점 계산
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        # 첫 프레임이면 centroid 등록
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # 기존 centroid들과 새로운 centroid간 거리 계산
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # 가장 가까운 것끼리 매칭
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

            # 매칭되지 않은 기존 객체는 사라진 것으로 간주
            unused_rows = set(range(0, D.shape[0])) - used_rows
            unused_cols = set(range(0, D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # 매칭되지 않은 새로운 centroid 등록
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

# ----------------------------
# 사용자 학습된 YOLOv5 best.pt 모델 로드
# ----------------------------
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # 사용자 모델 로드
model.conf = 0.5  # confidence threshold 설정
model.eval()  # 추론 모드 설정

# ----------------------------
# 메인 실행부: 카메라 캡처, 추론, 추적, 시각화
# ----------------------------
cap = cv2.VideoCapture(0)  # 기본 웹캠 사용
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)  # 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

tracker = CentroidTracker(max_disappeared=40)  # 객체 추적기 생성
frame_count = 0
last_boxes = []  # 최근 감지된 박스 저장용 리스트

while True:
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        break

    frame_count += 1

    if frame_count % 3 == 0:  # 매 3프레임마다 YOLO 추론
        results = model(frame)
        dets = results.xyxy[0].cpu().numpy()  # 결과를 numpy로 변환
        last_boxes = []
        for det in dets:
            x1, y1, x2, y2, conf, cls = det[:6]
            cls = int(cls)
            if conf < 0.5 or cls not in [0, 1, 2]:  # person, helmet, no-helmet 클래스만
                continue
            w, h = x2 - x1, y2 - y1
            if w < 30 or h < 30:  # 너무 작은 객체 무시
                continue
            last_boxes.append((int(x1), int(y1), int(x2), int(y2), cls))

    objects = tracker.update([(x1, y1, x2, y2) for (x1, y1, x2, y2, cls) in last_boxes])  # 추적기 업데이트

    # 추적 결과 시각화
    for (object_id, centroid) in objects.items():
        for (x1, y1, x2, y2, cls) in last_boxes:
            cX = int((x1 + x2) / 2)
            cY = int((y1 + y2) / 2)
            if abs(centroid[0] - cX) < 10 and abs(centroid[1] - cY) < 10:
                label = model.names[cls]  # 클래스 이름
                color = (0, 255, 0) if label == 'person' else (255, 255, 0) if label == 'helmet' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ID:{object_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                break

    cv2.imshow("YOLOv5 Custom + Centroid", frame)  # 결과 화면 표시
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()  # 자원 해제
