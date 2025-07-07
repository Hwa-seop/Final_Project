import os
import numpy as np
import json
from deepface import DeepFace

class FaceManager:
    def __init__(self, db_path="face_embeddings.json", model_name="Facenet"):
        self.db_path = db_path
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
            emb = DeepFace.represent(img_path=image_path, model_name=self.model_name)[0]['embedding']
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

    def verify_face(self, input_img_path, threshold=0.4):
        """입력 얼굴이 등록된 사용자 중 누구인지 확인"""
        try:
            input_emb = DeepFace.represent(img_path=input_img_path, model_name=self.model_name)[0]['embedding']
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
    