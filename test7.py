import cv2
import torch
from flask import Flask, render_template_string, Response

# 모델 로드 (GPU 사용 가능 시 GPU로)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', device=device)

app = Flask(__name__)
cap = cv2.VideoCapture(0)

frame_interval = 3  # YOLO 추론 간격 (3프레임마다)
frame_count = 0
last_dets = []

def gen_frames():
    global frame_count, last_dets
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 일정 간격마다 YOLO 추론
        if frame_count % frame_interval == 0:
            results = model(frame)
            last_dets = results.xyxy[0].cpu().numpy()

        # 마지막 추론 결과를 이용하여 박스 표시
        for det in last_dets:
            class_id = int(det[5])
            if class_id == 0:  # 사람
                x1, y1, x2, y2 = map(int, det[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # JPEG 압축 품질 설정 (기본값은 95)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_jpg = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_jpg + b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>실시간 감시 화면</title>
    </head>
    <body>
        <h2>실시간 감시 영상(YOLO 결과 포함)</h2>
        <img src="/video_feed" style="max-width: 100%;">
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
