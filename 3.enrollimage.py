import cv2
from deepface import DeepFace
import numpy as np
import pickle

cap = cv2.VideoCapture(0)
registered = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 얼굴 감지
    faces = DeepFace.extract_faces(img_path = frame, detector_backend = 'opencv', enforce_detection = False)
    for f in faces:
        area = f["facial_area"]
        x = area["x"]
        y = area["y"]
        w = area["w"]
        h = area["h"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, "Press S to Save", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Register Your Face (Press S)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and faces:
        # 얼굴 임베딩 추출 및 저장
        embedding = DeepFace.represent(img_path = frame, model_name = "Facenet")[0]["embedding"]
        with open("approved_face_deepface.pkl", "wb") as f:
            pickle.dump(np.array(embedding), f)
        print("등록 완료!")
        registered = True
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if registered:
    print("허용자 얼굴 등록이 성공적으로 저장되었습니다.")
else:
    print("등록 취소 또는 실패.")
