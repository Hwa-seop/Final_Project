import cv2
import torch
import numpy as np
from sort import Sort  # SORT 트래커 불러오기

# YOLOv5 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model.conf = 0.4  # confidence threshold

# SORT 트래커 초기화
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# 비디오 캡처
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    # SORT 입력 형식: [x1, y1, x2, y2, conf]
    person_dets = []
    for det in detections:
        if int(det[5]) == 0:  # person 클래스
            x1, y1, x2, y2, conf, cls = det
            person_dets.append([x1, y1, x2, y2, conf])

    # numpy 배열로 변환하여 SORT에 전달
    trackers = tracker.update(np.array(person_dets))

    # 결과 출력
    for d in trackers:
        x1, y1, x2, y2, track_id = d.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO + SORT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
