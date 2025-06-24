import cv2
import torch

# [1] 사각형 ROI 정의: (좌상단 x, y, 폭, 높이)
roi_x, roi_y, roi_w, roi_h = 100, 150, 200, 150  # 예시 좌표와 크기

def is_inside_roi(point, x, y, w, h):
    cx, cy = point
    return (x <= cx <= x+w) and (y <= cy <= y+h)

# [2] YOLOv5 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)
cv2.namedWindow("Danger Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Danger Detection", 1280, 960)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # [3] 위험구역(사각형) 그리기
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 3)

    # [4] YOLOv5 추론
    results = model(frame)
    persons = results.xyxy[0][results.xyxy[0][:,5]==0]  # 클래스 0: person

    danger = False

    for *box, conf, cls in persons.tolist():
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # [5] 중심점이 사각형 ROI에 들어갔는지 판정
        if is_inside_roi((cx, cy), roi_x, roi_y, roi_w, roi_h):
            cv2.putText(frame, "DANGER!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            danger = True
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    if danger:
        print("위험구역 진입: 경고 알림!")

    cv2.imshow("Danger Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
