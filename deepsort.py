import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# YOLOv5 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model.conf = 0.4  # confidence threshold

# DeepSORT 초기화
tracker = DeepSort(max_age=30)

# 비디오 캡처
cap = cv2.VideoCapture(0)

#실행 프레임 카운트
frame_count = 0
yolo_interval = 5  # 5프레임마다 YOLO 실행

while True:
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    if not ret:
        break

    frame_count += 1
    if frame_count % yolo_interval == 0:
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

    # DeepSORT에 입력할 bounding box 포맷: (xmin, ymin, width, height)
    person_dets = []
    for det in detections:
        if int(det[5]) == 0:  # 클래스가 person일 때만
            x1, y1, x2, y2, conf, cls = det
            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            person_dets.append(([bbox[0], bbox[1], bbox[2], bbox[3]], conf, 'person'))

    tracks = tracker.update_tracks(person_dets, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        cv2.rectangle(frame, (int(l), int(t)), (int(l + w), int(t + h)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO + DeepSORT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()