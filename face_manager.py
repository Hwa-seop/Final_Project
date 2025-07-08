import os
import shutil
import cv2
import numpy as np
import json
from deepface import DeepFace

# OpenCV haarcascade 경로 강제 지정
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
target_path = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

# 본인이 가지고 있는 haar 파일 경로
local_haar_path = "/home/hwaseop/final/haarcascade_frontalface_default.xml"

# 파일이 없으면 복사
if not os.path.exists(target_path):
    try:
        shutil.copy(local_haar_path, target_path)
        print(f"[✔] Haarcascade copied to {target_path}")
    except Exception as e:
        print(f"[X] Failed to copy haarcascade: {e}")

class FaceManager:
    def __init__(self, db_filename="face_embeddings.json", model_name="Facenet"):
        # 현재 파일 기준으로 절대 경로 생성
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(base_dir, db_filename)  # 절대 경로로 변경
        self.model_name = model_name
        self.face_db = self.load_db()
        
    def load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = f.read().strip()
                    if not data:
                        return {}
                    return json.loads(data)
            except Exception:
                return {}
        return {}

    def save_db(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.face_db, f)

    def register_face(self, username, image_path):
        """사용자 얼굴 등록"""
        try:
            emb = DeepFace.represent(
                img_path=image_path, 
                model_name=self.model_name,
                # enforce_detection=False,
                # detector_backend='skip'
                )[0]['embedding']
            self.face_db[username] = emb
            self.save_db()
            print(f"[✔] {username} 얼굴 등록 완료")
        except Exception as e:
            print(f"[X] 얼굴 등록 실패: {e}")

    def cosine_distance(self, vec1, vec2):
        """코사인 거리 계산 (값이 작을수록 유사)"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return 1 - (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def verify_face(self, img_path, threshold=0.4):
        """입력 얼굴이 등록된 사용자 중 누구인지 확인"""
        try:
            input_emb = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                # enforce_detection=False,
                # detector_backend='opencv'
                )[0]['embedding']
            best_match = None
            best_dist = float('inf')

            for username, stored_emb in self.face_db.items():
                dist = self.cosine_distance(input_emb, stored_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_match = username

            if best_dist <= threshold:
                confidence = round((1 - best_dist) * 100, 2)
                return True, best_match, confidence
            return False, None, 0.0

        except Exception as e:
            print(f"[X] 얼굴 인식 오류: {e}")
            return False, None, 0.0
    