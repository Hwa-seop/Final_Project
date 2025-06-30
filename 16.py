import cv2
import numpy as np
import torch
import time
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort


# 최근 경고 저장소
recent_alerts = []

def is_duplicate_alert(track_id, cooldown=10):
    current_time = time.time()
    for alert in recent_alerts:
        if alert['id'] == track_id and current_time - alert['time'] < cooldown:
            return True
    return False

def insert_alert(track_id, cx, cy):
    if is_duplicate_alert(track_id):
        return

    conn, cursor = None, None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = "INSERT INTO alerts (alert_time, coord_x, coord_y) VALUES (%s, %s, %s)"
        cursor.execute(sql, (now_str, cx, cy))
        conn.commit()

        recent_alerts.append({'id': track_id, 'cx': cx, 'cy': cy, 'time': time.time(), 'alert_time': now_str})
        recent_alerts[:] = [a for a in recent_alerts if time.time() - a['time'] < 60]

    except Exception as e:
        print(f"[DB ERROR] {e}")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

# 모델 및 트래커 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model.conf = 0.4
model.eval()

tracker = DeepSort(max_age=30)
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
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = frame.copy()
        results = model(frame)
        dets = results.xyxy[0].cpu().numpy()

        person_dets = []
        for det in dets:
            if int(det[5]) == 0:  # person
                x1, y1, x2, y2, conf, cls = det
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                person_dets.append(([bbox[0], bbox[1], bbox[2], bbox[3]], conf, 'person'))

        tracks = tracker.update_tracks(person_dets, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            cx, cy = int((l + r) / 2), int((t + b) / 2)

            inside = cv2.pointPolygonTest(zone_poly, (cx, cy), False) if zone_locked else -1
            if inside >= 0:
                cv2.rectangle(output, (int(l), int(t)), (int(r), int(b)), (0, 0, 255), 2)
                cv2.putText(output, f'ID: {track_id} DANGER!', (int(l), int(t)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                insert_alert(track_id, cx, cy)
            else:
                cv2.rectangle(output, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
                cv2.putText(output, f'ID: {track_id}', (int(l), int(t)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.circle(output, (cx, cy), 4, (255, 0, 0), -1)

        if zone_locked and zone_poly is not None:
            cv2.polylines(output, [zone_poly], isClosed=True, color=(0,0,255), thickness=3)

        cv2.imshow("Security Alert", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    roi_setup()
    main_loop()