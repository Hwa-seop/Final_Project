import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template_string
import threading
import time

# YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model.eval()  # type: ignore

cap = cv2.VideoCapture(0)
app = Flask(__name__)

# ROI 관련 변수
roi_points = []
zone_locked = False
zone_poly = None

# 공유 변수
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
        elif event == cv2.EVENT_MBUTTONDOWN and len(roi_points) >= 3:
            zone_poly = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))
            zone_locked = True
            print("[INFO] ROI 설정 완료.")

def roi_setup():
    global zone_locked, zone_poly

    print("[INFO] 마우스로 ROI 다각형을 설정하세요 (좌클릭: 점 추가, 우클릭: 삭제, 휠클릭: 확정)")
    cv2.namedWindow("ROI 설정")
    cv2.setMouseCallback("ROI 설정", mouse_callback)

    while not zone_locked:
        ret, frame = cap.read()
        if not ret:
            break
        output = frame.copy()

        for pt in roi_points:
            cv2.circle(output, pt, 5, (255, 0, 0), -1)
        if len(roi_points) > 1:
            cv2.polylines(output, [np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))],
                          isClosed=False, color=(0, 255, 255), thickness=2)
        cv2.putText(output, "왼쪽=점추가, 오른쪽=삭제, 휠클릭=완료", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("ROI 설정", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow("ROI 설정")

def yolo_streaming_loop():
    global shared_frame, stop_flag
    frame_count = 0
    yolo_interval = 10
    last_dets = []

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break

        output = frame.copy()
        frame_count += 1

        if frame_count % yolo_interval == 0:
            results = model(frame)
            last_dets = results.xyxy[0].cpu().numpy()
        dets = last_dets

        alert = False
        for det in dets:
            if int(det[5]) == 0:  # person
                x1, y1, x2, y2, conf = det[:5]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                inside = cv2.pointPolygonTest(zone_poly, (cx, cy), False)
                if inside >= 0:
                    alert = True
                    print(f"[ALERT] 사람 감지: 좌표=({cx}, {cy})")  # ✅ 올바른 위치
                    cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(output, "DANGER!", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(output, (cx, cy), 3, (255, 0, 0), -1)

        if zone_poly is not None:
            cv2.polylines(output, [zone_poly], isClosed=True,
                          color=(0, 0, 255) if alert else (0, 255, 0), thickness=2)

        with frame_lock:
            shared_frame = output.copy()

        time.sleep(0.01)


def gen_frames():
    global shared_frame
    while not stop_flag:
        with frame_lock:
            if shared_frame is None:
                continue
            frame = shared_frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template_string("""
    <html><head><title>ROI 감시</title></head>
    <body>
        <h2>ROI 감시 스트리밍</h2>
        <img src="/video_feed">
    </body></html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# === 실행 ===
if __name__ == '__main__':
    roi_setup()  # 1단계: ROI 마우스로 설정
    threading.Thread(target=yolo_streaming_loop, daemon=True).start()  # 2단계: YOLO 감지
    print("[INFO] Flask 스트리밍 시작: http://<서버IP>:5000")
    app.run(host='0.0.0.0', port=5000)
