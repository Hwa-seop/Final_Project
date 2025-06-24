import cv2
import numpy as np
import torch

# 빨강 HSV
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# YOLOv5 모델
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)
alert_active = False
blink_state = False
blink_counter = 0
zone_locked = False
locked_zone_poly = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 감지 영역 설정(다각형 ROI)
    if not zone_locked:
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            zone_poly = cv2.approxPolyDP(largest, 0.02 * cv2.arcLength(largest, True), True)
        else:
            zone_poly = None
    else:
        zone_poly = locked_zone_poly

    alert_active = False

    # === YOLO로 사람 검출 및 ROI 침입 확인 ===
    danger_this_frame = False  # 이 프레임에서 경고 출력 여부(중복 방지)
    if zone_poly is not None and zone_locked:
        # YOLOv5로 추론
        results = model(frame)
        dets = results.xyxy[0].cpu().numpy()
        for det in dets:
            if int(det[5]) == 0:  # person
                x1, y1, x2, y2, conf = det[:5]
                cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
                inside = cv2.pointPolygonTest(zone_poly, (cx, cy), False)
                if inside >= 0:
                    alert_active = True
                    danger_this_frame = True
                    # 화면 경고 표시
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
                cv2.drawContours(output_frame, [zone_poly], -1, (0, 0, 255), 3)
                cv2.putText(output_frame, "WARNING!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                cv2.drawContours(output_frame, [zone_poly], -1, (0, 255, 0), 2)
        else:
            cv2.drawContours(output_frame, [zone_poly], -1, (0, 255, 0), 2)
    else:
        cv2.putText(output_frame, "영역이 감지되지 않았습니다", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # === 콘솔에 WARNING 메시지 출력 ===
    if danger_this_frame:
        print("[WARNING] 사람 침입 감지!")

    cv2.imshow("Security Alert", output_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("h"):
        if zone_poly is not None:
            locked_zone_poly = zone_poly
            zone_locked = True
            print("[INFO] 감지 영역이 고정되었습니다.")
    elif key == ord("r"):
        zone_locked = False
        locked_zone_poly = None
        print("[INFO] 감지 영역이 재설정되었습니다.")

cap.release()
cv2.destroyAllWindows()
