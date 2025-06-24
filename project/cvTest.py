import cv2
import numpy as np
from datetime import datetime

# 붉은 계열 HSV 범위
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

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

    # 감지 영역 설정
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

    if zone_poly is not None:
        # 프레임 전처리
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if 'prev_gray' in locals():
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            motion_cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            alert_active = False
            for c in motion_cnts:
                if cv2.contourArea(c) < 500:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                center = (x + w // 2, y + h // 2)
                inside = cv2.pointPolygonTest(zone_poly, center, False)
                if inside >= 0 and zone_locked:  # ✅ 고정된 상태에서만 경고 허용
                    alert_active = True
                    break

        prev_gray = gray

        # 경고 시 깜빡이면서 표시 (고정 상태일 때만)
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
