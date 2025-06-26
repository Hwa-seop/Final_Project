import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template_string
import threading
import time
import os
# YOLOv5 모델을 PyTorch Hub에서 불러옴 (가볍고 빠른 'n'버전 사용)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model.eval() # type: ignore

cap = cv2.VideoCapture(0)
app = Flask(__name__)

# -------------------- ROI 설정 관련 변수 --------------------
roi_points = []
zone_locked = False
zone_poly = None

# -------------------- 스레드 & 공유 변수 --------------------
stop_flag = False
shared_frame = None
frame_lock = threading.Lock()

# -------------------- 마우스 이벤트 콜백 (ROI 설정용) --------------------
def mouse_callback(event, x, y, flags, param):
    global roi_points, zone_locked, zone_poly
    if not zone_locked:
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))   # 왼쪽 클릭: 점 추가
        elif event == cv2.EVENT_RBUTTONDOWN and roi_points:
            roi_points.pop()            # 오른쪽 클릭: 마지막 점 삭제

# -------------------- ROI 다각형 그리기 및 확정 함수 --------------------
import os

def roi_setup():
    global zone_locked, zone_poly, stop_flag
    print("[INFO] 마우스로 ROI 다각형을 설정하세요 (좌클릭: 점 추가, 우클릭: 삭제, Enter/Space: 확정, Q/q: 종료)")
    cv2.namedWindow("ROI 설정", cv2.WINDOW_NORMAL)
    time.sleep(0.1)  # 안정화를 위한 짧은 대기
    try:
        cv2.setMouseCallback("ROI 설정", mouse_callback)
    except cv2.error as e:
        print(f"[WARN] setMouseCallback 실패: {e}")

    while not zone_locked and not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break
        output = frame.copy()
        for pt in roi_points:
            cv2.circle(output, pt, 5, (255, 0, 0), -1)
        if len(roi_points) > 1:
            cv2.polylines(output, [np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))],
                          isClosed=False, color=(0, 255, 255), thickness=2)
        cv2.putText(output, "L-click=add, R-click=del, Enter/Space=done, Q/q=quit", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("ROI 설정", output)
        key = cv2.waitKeyEx(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("[INFO] 사용자가 q/Q를 눌러 종료합니다.")
            cv2.destroyAllWindows()
            cap.release()
            os._exit(0)
        elif (key == 13 or key == 32) and len(roi_points) >= 3:
            zone_poly = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))
            zone_locked = True
            print("[INFO] ROI 설정 완료.")

    cv2.destroyWindow("ROI 설정")

# -------------------- YOLOv5 사람 감지 & 경계 알림 (스레드) --------------------
def yolo_streaming_loop():
    global shared_frame, stop_flag
    frame_count = 0
    yolo_interval = 10   # YOLO를 10프레임마다 1회 실행 (속도 최적화)
    last_dets = []

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break

        output = frame.copy()
        frame_count += 1

        # YOLO 추론 실행 (10프레임마다 1회)
        if frame_count % yolo_interval == 0:
            results = model(frame)
            last_dets = results.xyxy[0].cpu().numpy()
        dets = last_dets

        alert = False
        for det in dets:
            if int(det[5]) == 0:  # 클래스 0: person(사람)
                x1, y1, x2, y2, conf = det[:5]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                inside = cv2.pointPolygonTest(zone_poly, (cx, cy), False)
                if inside >= 0:
                    alert = True
                    print(f"[ALERT] 사람 감지: 좌표=({cx}, {cy})")
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

# -------------------- Flask용 프레임 스트리밍 제너레이터 --------------------
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

# -------------------- Flask 웹 라우팅 --------------------
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

# -------------------- 메인 실행부 --------------------
if __name__ == '__main__':
    roi_setup()  # 1단계: ROI 마우스로 설정
    if not stop_flag:
        threading.Thread(target=yolo_streaming_loop, daemon=True).start()  # 2단계: YOLO 감지 (스레드)
        print("[INFO] Flask 스트리밍 시작: http://<서버IP>:5000")
        try:
            app.run(host='0.0.0.0', port=5000)
        except KeyboardInterrupt:
            print("[INFO] 수동 종료 감지.")
    # 종료 처리
    stop_flag = True
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 프로그램을 종료합니다.")
