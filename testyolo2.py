import cv2  # OpenCV 라이브러리 임포트 (영상 처리)
import numpy as np  # NumPy 라이브러리 임포트 (행렬 계산)

# 클래스 이름 텍스트 파일에서 불러오기
def read_class_names(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]  # 줄바꿈 제거 후 리스트로 반환

# 경로 설정: ONNX 모델과 클래스 이름 파일
model_path = "/home/lws/kulws2025/kubig2025/final_project/yolov5/helmet-yolo/yolov5s_helmet3/weights/best.onnx"
class_names_file = "/home/lws/kulws2025/kubig2025/final_project/yolov5/helmet-yolo/yolov5s_helmet3/weights/classes.txt"
class_names = read_class_names(class_names_file)  # 클래스 이름 리스트 로드

# ONNX 모델 로드
net = cv2.dnn.readNetFromONNX(model_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # OpenCV 백엔드 사용
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # CPU에서 실행

# 웹캠 열기 (0: 기본 카메라)
cap = cv2.VideoCapture(0)
cv2.namedWindow("YOLOv5 Detection", cv2.WINDOW_NORMAL)  # 출력 창 이름 설정
cv2.resizeWindow("YOLOv5 Detection", 1280, 720)  # 창 크기 설정

# 프레임 반복 처리
while True:
    ret, frame = cap.read()  # 프레임 캡처
    if not ret:
        break  # 프레임 읽기 실패 시 종료

    height, width = frame.shape[:2]  # 프레임의 높이와 너비 추출

    # 이미지 전처리: blob 형태로 변환 (정규화, 크기 조정, BGR->RGB)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)  # 모델 입력 설정
    outputs = net.forward(net.getUnconnectedOutLayersNames())  # 추론 수행
    output = outputs[0]  # 결과 가져오기 (배치 크기 1 기준)

    # 탐지된 사람과 헬멧 정보 저장 리스트
    persons, helmets = [], []
    person_scores, helmet_scores = [], []

    # 모든 탐지 결과 반복 처리
    for i in range(output.shape[1]):
        data = output[0, i]
        conf = data[4]  # 객체 존재 확률 (objectness score)
        if conf < 0.5:
            continue  # 신뢰도 낮으면 무시

        class_scores = data[5:]  # 클래스별 확률
        class_id = np.argmax(class_scores)  # 가장 높은 클래스 인덱스
        score = class_scores[class_id]  # 해당 클래스의 확률
        if conf * score < 0.5:
            continue  # 클래스 신뢰도 낮으면 무시

        # 바운딩 박스 좌표 변환 (YOLO 형식 → 픽셀 좌표)
        cx, cy, w, h = data[0:4]
        x = int((cx - w / 2) * width)
        y = int((cy - h / 2) * height)
        w = int(w * width)
        h = int(h * height)
        box = [x, y, w, h]
        label = class_names[class_id]  # 클래스 이름 가져오기

        # 클래스에 따라 분류
        if label == "person":
            persons.append(box)
            person_scores.append(conf * score)
        elif label == "helmet":
            helmets.append(box)
            helmet_scores.append(conf * score)

    # 시각화: 사람마다 헬멧 착용 여부 확인
    for i, (px, py, pw, ph) in enumerate(persons):
        person_rect = [px, py, px+pw, py+ph]  # 사람의 바운딩 박스
        has_helmet = False  # 초기값: 헬멧 미착용

        # 사람 바운딩 박스와 헬멧 박스 간 교차 여부 확인
        for j, (hx, hy, hw, hh) in enumerate(helmets):
            helmet_rect = [hx, hy, hx+hw, hy+hh]
            ix1, iy1 = max(person_rect[0], helmet_rect[0]), max(person_rect[1], helmet_rect[1])
            ix2, iy2 = min(person_rect[2], helmet_rect[2]), min(person_rect[3], helmet_rect[3])
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            if iw * ih > 0:  # 교차 면적 존재하면 착용으로 간주
                has_helmet = True

                # 헬멧 박스 그리기 (파란색)
                helmet_score = helmet_scores[j]
                cv2.rectangle(frame, (hx, hy), (hx+hw, hy+hh), (255, 0, 0), 2)
                cv2.putText(frame, f"helmet: {helmet_score:.2f}", (hx, hy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 사람 박스 색상 설정: 초록(착용), 빨강(미착용)
        color = (0, 255, 0) if has_helmet else (0, 0, 255)
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), color, 2)
        cv2.putText(frame, f"person: {person_scores[i]:.2f}", (px, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 결과 영상 출력
    cv2.imshow("YOLOv5 Detection", frame)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) == 27:
        break

# 종료 처리
cap.release()  # 카메라 해제
cv2.destroyAllWindows()  # 모든 창 닫기
