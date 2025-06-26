import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template_string
import threading
import time

# YOLOv5n 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model.eval()  # 추론 모드 설정

# 웹캠 캡처 객체
cap = cv2.VideoCapture(0)

# Flask 웹 앱 초기화
app = Flask(__name__)

# ROI 설정 관련 전역 변수
roi_points = []        # 마우스로 선택된 ROI 점들
zone_locked = False    # ROI 설정 확정 여부
zone_poly = None       # ROI 다각형 영역

# 멀티스레드 공유 변수
stop_flag = False
shared_frame = None
frame_lock = threading.Lock()

# 마우스 콜백 함수 - ROI 다각형 점 선택
def mouse_callback(event, x, y, flags, param):
    global roi_points, zone_locked
    if not zone_locked:
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and roi_points:
            roi_points.pop()

# ROI 다각형 설정 루프
def roi_setup():
    global zone_locked, zone_poly, stop_flag

    print("[INFO] Draw ROI polygon. Left: add, Right: remove, Enter/Space: confirm, q: quit")
    cv2.namedWindow("Security Alert", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Security Alert", mouse_callback)

    while not zone_locked and not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break
        output = frame.copy()

        # 현재 ROI 점 표시
        for pt in roi_points:
            cv2.circle(output, pt, 5, (255, 0, 0), -1)
        if len(roi_points) > 1:
            cv2.polylines(output, [np.array(roi_points, np.int32).reshape((-1,1,2))],
                          False, (0, 255, 255), 2)

        # 안내 메시지 출력
        cv2.putText(output, "Left: add point, Right: remove", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(output, "Enter/Space: confirm ROI, q: quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Security Alert", output)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            stop_flag = True
            break
        elif key in [13, 32] and len(roi_points) >= 3:
            zone_poly = np.array(roi_points, dtype=np.int32).reshape((-1,1,2))
            zone_locked = True
            print("[INFO] ROI confirmed.")

    cv2.destroyAllWindows()

# YOLO 탐지 및 ROI 경고 표시 루프 (백그라운드 쓰레드에서 실행)
def yolo_loop():
    global shared_frame, stop_flag, zone_locked, zone_poly

    frame_count = 0
    yolo_interval = 10  # 10프레임마다 YOLO 실행
    last_detections = []

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break

        output = frame.copy()
        frame_count += 1

        if zone_locked and zone_poly is not None:
            if frame_count % yolo_interval == 0:
                results = model(frame)
                last_detections = results.xyxy[0].cpu().numpy()

            alert_active = False
            for det in last_detections:
                if int(det[5]) == 0:  # 사람 클래스
                    x1, y1, x2, y2, conf = det[:5]
                    cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
                    inside = cv2.pointPolygonTest(zone_poly, (cx, cy), False)

                    # ROI 내부에 있을 경우 경고
                    color = (0, 0, 255) if inside >= 0 else (0, 255, 0)
                    cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    if inside >= 0:
                        alert_active = True
                        cv2.putText(output, "DANGER!", (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv2.circle(output, (cx, cy), 4, (255, 0, 0), -1)

            poly_color = (0,0,255) if alert_active else (0,255,0)
            cv2.polylines(output, [zone_poly], True, poly_color, 3)
            if alert_active:
                cv2.putText(output, "WARNING!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        else:
            # ROI 미설정 시 사용자 가이드 표시
            for pt in roi_points:
                cv2.circle(output, pt, 5, (255, 0, 0), -1)
            if len(roi_points) > 1:
                cv2.polylines(output, [np.array(roi_points, np.int32).reshape((-1,1,2))],
                              False, (0, 255, 255), 2)
            cv2.putText(output, "Draw ROI and press Enter", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # 공유 프레임 업데이트
        with frame_lock:
            shared_frame = output.copy()

        time.sleep(0.01)  # CPU 부하 방지

# Flask용 프레임 제너레이터
def gen_frames():
    global shared_frame, stop_flag
    while not stop_flag:
        with frame_lock:
            if shared_frame is None:
                continue
            frame = shared_frame.copy()

        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.03)

# 기본 웹페이지 라우팅
@app.route('/')
def index():
    return render_template_string("""
    <html><head><title>Security Alert</title></head>
    <body>
        <h2>Security Alert Camera Stream</h2>
        <img src="/video_feed" width="720" />
    </body></html>
    """)

# 영상 스트리밍 라우트
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 메인 실행
if __name__ == '__main__':
    roi_setup()  # ROI 다각형 먼저 설정
    if not stop_flag:
        threading.Thread(target=yolo_loop, daemon=True).start()  # YOLO 루프 시작
        print("[INFO] Starting Flask server at http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000)

    # 종료 시 리소스 해제
    cap.release()
    cv2.destroyAllWindows()
