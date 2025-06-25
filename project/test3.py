import cv2
import numpy as np
import torch

# YOLOv5n 모델 사용
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# 캡처 및 감지 상태 체크 변수
cap = cv2.VideoCapture(0)
alert_active = False
blink_state = False
blink_counter = 0

# 영역 지정 변수
roi_points = []
zone_locked = False
zone_poly = None

def mouse_callback(event, x, y, flags, param):
    global roi_points, zone_locked, zone_poly
    if not zone_locked:
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:  # 오른쪽 클릭 시 마지막 점 삭제
            if roi_points:
                roi_points.pop()
        elif event == cv2.EVENT_MBUTTONDOWN:  # 가운데 버튼(휠) 클릭 시 확정
            if len(roi_points) >= 3:
                zone_poly = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))
                zone_locked = True
                print("[INFO] Area fixed")

cv2.namedWindow("Security Alert")
cv2.setMouseCallback("Security Alert", mouse_callback)

frame_count = 0
yolo_interval = 60
last_dets = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = frame.copy()

    # ROI 폴리곤 지정 중이면 점/선 미리 그리기
    if not zone_locked and roi_points:
        for pt in roi_points:
            cv2.circle(output_frame, pt, 4, (255, 0, 0), -1)
        if len(roi_points) > 1:
            cv2.polylines(output_frame, [np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))],
                          isClosed=False, color=(0, 255, 255), thickness=2)
        cv2.putText(output_frame, "Left : Add | Right = Delete | Wheel Click = Done", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    alert_active = False
    danger_this_frame = False

    frame_count += 1

    # ROI 다각형이 확정된 뒤에만 YOLO 실행
    if zone_locked and zone_poly is not None:
        if frame_count % yolo_interval == 0:
            results = model(frame)
            dets = results.xyxy[0].cpu().numpy()
            last_dets = dets
        else:
            dets = last_dets

        for det in dets:
            if int(det[5]) == 0:  # person
                x1, y1, x2, y2, conf = det[:5]
                cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
                inside = cv2.pointPolygonTest(zone_poly, (cx, cy), False)
                if inside >= 0:
                    alert_active = True
                    danger_this_frame = True
                    cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                    cv2.putText(output_frame, "DANGER!", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                else:
                    cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.circle(output_frame, (cx, cy), 4, (255, 0, 0), -1)

    # ==== ROI 다각형 표시 및 경고 ====
    if zone_poly is not None:
        if zone_locked and alert_active:
            blink_counter += 1
            if blink_counter % 10 == 0:
                blink_state = not blink_state

            if blink_state:
                cv2.polylines(output_frame, [zone_poly], isClosed=True, color=(0, 0, 255), thickness=3)
                cv2.putText(output_frame, "WARNING!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                cv2.polylines(output_frame, [zone_poly], isClosed=True, color=(0, 255, 0), thickness=2)
        else:
            cv2.polylines(output_frame, [zone_poly], isClosed=True, color=(0, 255, 0), thickness=2)

    if not zone_locked and not roi_points:
        cv2.putText(output_frame, "Please Draw Area", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    if danger_this_frame:
        print("[WARNING] Access Detected")

    cv2.imshow("Security Alert", output_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("r"):
        zone_locked = False
        roi_points = []
        zone_poly = None
        print("[INFO] Area reset")

cap.release()
cv2.destroyAllWindows()
