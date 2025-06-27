import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template_string
import threading
import time
import mysql.connector
from datetime import datetime

# MySQL 연결 설정
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'qwe123',  # 실제 비밀번호로 변경
    'database': 'safety_monitoring'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

# 중복 저장 방지를 위한 최근 경고 메모리 저장소
recent_alerts = []

def is_duplicate_alert(cx, cy, threshold=30, cooldown=10):
    current_time = time.time()
    for alert in recent_alerts:
        dist = ((cx - alert['cx'])**2 + (cy - alert['cy'])**2)**0.5
        if dist < threshold and current_time - alert['time'] < cooldown:
            return True
    return False

def insert_alert(cx, cy):
    if is_duplicate_alert(cx, cy):
        return  # 중복 경고 저장 안함
    
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = "INSERT INTO alerts (alert_time, coord_x, coord_y) VALUES (%s, %s, %s)"
        cursor.execute(sql, (now_str, cx, cy))
        conn.commit()
        print(f"[DB] Alert inserted at {now_str} coords=({cx}, {cy})")

        # 메모리에도 저장 (최근 60초치만 유지)
        recent_alerts.append({'cx': cx, 'cy': cy, 'time': time.time(), 'alert_time': now_str})
        recent_alerts[:] = [a for a in recent_alerts if time.time() - a['time'] < 60]

    except Exception as e:
        print(f"[DB ERROR] {e}")
    finally:
        try:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        except Exception as e:
            print(f"[DB ERROR] Error closing DB connection: {e}")

# YOLOv5n model load
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model.eval()

cap = cv2.VideoCapture(0)
app = Flask(__name__)

roi_points = []
zone_locked = False
zone_poly = None

stop_flag = False
shared_frame = None
frame_lock = threading.Lock()

def mouse_callback(event, x, y, flags, param):
    global roi_points, zone_locked, zone_poly
    if not zone_locked:
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and roi_points:
            roi_points.pop()

def roi_setup():
    global zone_locked, zone_poly, stop_flag

    print("[INFO] Draw ROI polygon with mouse. Left click: add point, Right click: remove point, Enter/Space: confirm, q: quit")
    cv2.namedWindow("Security Alert", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Security Alert", mouse_callback)

    while not zone_locked and not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break
        output = frame.copy()

        for pt in roi_points:
            cv2.circle(output, pt, 5, (255, 0, 0), -1)
        if len(roi_points) > 1:
            cv2.polylines(output, [np.array(roi_points, dtype=np.int32).reshape((-1,1,2))],
                          isClosed=False, color=(0, 255, 255), thickness=2)
        cv2.putText(output, "Left click: add point, Right click: remove point", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(output, "Enter/Space: confirm ROI, q: quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Security Alert", output)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[INFO] Quit by user.")
            stop_flag = True
            break
        elif (key == 13 or key == 32) and len(roi_points) >= 3:
            zone_poly = np.array(roi_points, dtype=np.int32).reshape((-1,1,2))
            zone_locked = True
            print("[INFO] ROI polygon confirmed.")

    cv2.destroyAllWindows()

def yolo_loop():
    global shared_frame, stop_flag, zone_locked, zone_poly

    frame_count = 0
    yolo_interval = 5
    last_detections = []

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break

        output = frame.copy()
        frame_count += 1

        if zone_locked and zone_poly is not None:
            if frame_count % yolo_interval == 0:
                try:
                    results = model(frame)
                    last_detections = results.xyxy[0].cpu().numpy()
                except Exception as e:
                    print(f"[YOLO ERROR] {e}")

            dets = last_detections
            alert_active = False

            for det in dets:
                if int(det[5]) == 0:  # person class
                    x1, y1, x2, y2, conf = det[:5]
                    cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
                    inside = cv2.pointPolygonTest(zone_poly, (cx, cy), False)
                    if inside >= 0:
                        alert_active = True
                        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                        cv2.putText(output, "DANGER!", (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                        try:
                            insert_alert(cx, cy)
                        except Exception as e:
                            print(f"[DB ERROR in loop] {e}")
                    else:
                        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.circle(output, (cx, cy), 4, (255, 0, 0), -1)

            if alert_active:
                cv2.polylines(output, [zone_poly], isClosed=True, color=(0,0,255), thickness=3)
                cv2.putText(output, "WARNING!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                cv2.polylines(output, [zone_poly], isClosed=True, color=(0,255,0), thickness=2)
        else:
            if roi_points:
                for pt in roi_points:
                    cv2.circle(output, pt, 5, (255, 0, 0), -1)
                if len(roi_points) > 1:
                    cv2.polylines(output, [np.array(roi_points, dtype=np.int32).reshape((-1,1,2))],
                                  isClosed=False, color=(0, 255, 255), thickness=2)
                cv2.putText(output, "Draw ROI polygon and confirm with Enter/Space", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        with frame_lock:
            shared_frame = output.copy()

        time.sleep(0.01)

def gen_frames():
    global shared_frame, stop_flag
    while not stop_flag:
        with frame_lock:
            if shared_frame is None:
                continue
            frame = shared_frame.copy()

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.03)

# 실시간 최근 경고 조회용 API
@app.route('/alerts')
def get_alerts():
    return {'alerts': recent_alerts}

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Security Alert</title>
        <script>
            async function fetchAlerts() {
                const response = await fetch('/alerts');
                const data = await response.json();
                let list = document.getElementById('alert-list');
                list.innerHTML = '';
                data.alerts.forEach(alert => {
                    let item = document.createElement('li');
                    item.textContent = alert.alert_time + ' - Coordinates: (' + alert.cx + ',' + alert.cy + ')';
                    list.appendChild(item);
                });
            }
            setInterval(fetchAlerts, 3000);
            window.onload = fetchAlerts;
        </script>
    </head>
    <body>
        <h2>Security Alert Camera Stream</h2>
        <img src="/video_feed" width="720" />
        <h3>Recent Alerts</h3>
        <ul id="alert-list"></ul>
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    roi_setup()
    if not stop_flag:
        t = threading.Thread(target=yolo_loop, daemon=True)
        t.start()
        print("[INFO] Starting Flask server at http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000)
    cap.release()
    cv2.destroyAllWindows()
