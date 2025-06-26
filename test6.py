import cv2
import numpy as np
import torch
from flask import Flask, render_template_string, Response

# YOLOv5n 모델
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
# 필요하다면 ROI 관련 변수 선언 (ex. zone_poly, zone_locked 등)

app = Flask(__name__)

cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- YOLO/ROI 관련 처리 (아래 예시는 "사람"만 빨간 박스 표시) ---
        results = model(frame)
        dets = results.xyxy[0].cpu().numpy()
        for det in dets:
            if int(det[5]) == 0:  # person
                x1, y1, x2, y2 = map(int, det[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # -------------------------------------------------------------
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_jpg = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_jpg + b'\r\n')
        
        # 결과 영상 표시
        cv2.imshow("YOLOv5 디버그 영상", frame)

        # 'q' 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>실시간 감시 화면</title>
    </head>
    <body>
        <h2>실시간 감시 영상(YOLO 결과 포함)</h2>
        <img src="/video_feed">
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)