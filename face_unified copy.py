import os
# TensorFlow 경고 메시지 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import cv2
import sys
import numpy as np
from face_manager import FaceManager

# Haar Cascade 로드 (전역)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # 얼굴 인식에 사용되는 Haar Cascade 분류기

class FaceUnified:
    def __init__(self, threshold=0.4):
        self.face_manager = FaceManager()  # FaceManager 인스턴스 생성
        self.cap = None
        self.threshold = threshold
        
    def start_camera(self):
        """웹캠 시작"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[X] Cannot open webcam.")
            return False
        return True
    
    def stop_camera(self):
        """웹캠 종료"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def draw_text_with_background(self, frame, text, position, font_scale=0.7, 
                                 text_color=(255, 255, 255), bg_color=(0, 0, 0)):
        """텍스트에 배경을 추가하여 그리기"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        
        # 텍스트 크기 계산
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 배경 사각형 그리기
        x, y = position
        cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width + 10, y + 10), bg_color, -1)
        
        # 텍스트 그리기
        cv2.putText(frame, text, (x + 5, y - 5), font, font_scale, text_color, thickness)
    
    def capture_face(self, save_path):
        """얼굴 촬영 및 저장"""
        if not self.start_camera():
            return False
            
        print("[INFO] Press SPACE to capture, ESC to cancel.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[X] Cannot read frame.")
                break
                
            # 화면에 안내 텍스트 표시
            cv2.putText(frame, "Press SPACE to capture, ESC to cancel", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("[CANCEL] Capture cancelled.")
                break
            elif key == 32:  # 스페이스바
                # 이미지 저장
                cv2.imwrite(save_path, frame)
                print(f"[OK] Face saved to {save_path}")
                break
        
        self.stop_camera()
        return True
    
    def register_face_with_camera(self, username):
        """카메라로 얼굴 촬영 후 등록"""
        # 저장할 이미지 경로
        save_path = f"faces/{username}.jpg"
        
        # faces 폴더가 없으면 생성
        os.makedirs("faces", exist_ok=True)
        
        print(f"[INFO] Capturing face for {username}...")
        
        # 얼굴 촬영
        if self.capture_face(save_path):
            # 얼굴 등록
            self.face_manager.register_face(username, save_path)
            return True
        return False
    
    def is_db_empty(self):
        db = self.face_manager.face_db
        if not db or not isinstance(db, dict):
            return True
        for k, v in db.items():
            if str(k).strip() and v:
                return False
        return True

    def realtime_face_recognition(self):
        if self.is_db_empty():
            print("\n[Registered Users]")
            print("No data")
            sys.stdout.flush()
            input("Press Enter to continue...")
            return
        if not self.start_camera():
            print("[X] Webcam could not be opened. Please check your camera.")
            return
        print("[INFO] Real-time face recognition started.")
        print("[INFO] Press ESC to exit.")
        print(f"[THRESHOLD] {self.threshold} (Recognized if below this value)")
        frame_shown = False
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[X] Cannot read frame from webcam.")
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "[X] Cannot read frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow('Real-time Face Recognition', blank)
                cv2.waitKey(1500)
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                # 얼굴 영역 추출
                face_img = frame[y:y+h, x:x+w]
                temp_face_path = "temp_face_crop.jpg"
                cv2.imwrite(temp_face_path, face_img)
                try:
                    found, username, confidence = self.face_manager.verify_face(temp_face_path, self.threshold)
                    if found:
                        label = f"{username}"
                        color = (0, 255, 0)
                    else:
                        label = "Not recognized"
                        color = (0, 0, 255)
                except Exception as e:
                    label = "Error"
                    color = (0, 255, 255)
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)
                # 네모 박스 및 라벨 표시
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y-25), (x+w, y), color, -1)
                cv2.putText(frame, label, (x+5, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            self.draw_text_with_background(frame, "ESC: Exit", (10, frame.shape[0] - 30), font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0))
            cv2.imshow('Real-time Face Recognition', frame)
            frame_shown = True
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[EXIT] Real-time recognition stopped.")
                break
        if not frame_shown:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "[X] No frame to show", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Real-time Face Recognition', blank)
            cv2.waitKey(1500)
        self.stop_camera()
    
    def detailed_face_analysis(self):
        if self.is_db_empty():
            print("\n[Registered Users]")
            print("No data")
            sys.stdout.flush()
            input("Press Enter to continue...")
            return
        if not self.start_camera():
            print("[X] Webcam could not be opened. Please check your camera.")
            return
        print("[INFO] Face analysis started.")
        print("[INFO] Press ESC to exit.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[X] Cannot read frame.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                temp_face_path = "temp_face_crop.jpg"
                cv2.imwrite(temp_face_path, face_img)
                try:
                    from deepface import DeepFace
                    input_emb = DeepFace.represent(img_path=temp_face_path, model_name=self.face_manager.model_name)[0]['embedding']
                    matches = []
                    for username, stored_emb in self.face_manager.face_db.items():
                        dist = self.face_manager.cosine_distance(input_emb, stored_emb)
                        confidence = (1 - dist) * 100
                        matches.append((username, confidence, dist))
                    matches.sort(key=lambda x: x[1], reverse=True)
                    if matches and matches[0][1] >= 50:
                        label = f"{matches[0][0]}"
                        color = (0, 255, 0)
                    else:
                        label = "Not recognized"
                        color = (0, 0, 255)
                except Exception as e:
                    label = "Error"
                    color = (0, 255, 255)
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y-25), (x+w, y), color, -1)
                cv2.putText(frame, label, (x+5, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            self.draw_text_with_background(frame, "ESC: Exit", (10, frame.shape[0] - 30), font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0))
            cv2.imshow('Face Analysis', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[EXIT] Face analysis stopped.")
                break
        self.stop_camera()
    
    def show_registered_users(self):
        print("\n[Registered Users]")
        if self.is_db_empty():
            print("No data")
        else:
            for name in self.face_manager.face_db.keys():
                print(f"- {name}")
        sys.stdout.flush()
        input("Press Enter to continue...")
    
    def delete_user(self):
        self.show_registered_users()
        username = input("Enter the user name to delete: ").strip()
        if not username:
            print("[X] Please enter a name.")
            return
        if username not in self.face_manager.face_db:
            print(f"[X] User '{username}' not found.")
            return
        # 이미지 파일 삭제
        img_path = f"faces/{username}.jpg"
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"[OK] Deleted image file: {img_path}")
        # DB에서 삭제
        del self.face_manager.face_db[username]
        self.face_manager.save_db()
        print(f"[OK] Deleted user '{username}' from database.")
    
    def show_menu(self):
        """메뉴 표시"""
        print("\n" + "="*50)
        print("🎯 Unified Face Recognition Program")
        print("="*50)
        print("1. Register Face")
        print("2. Real-time Recognition")
        print("3. Face Analysis")
        print("4. Show Registered Users")
        print("5. Delete User Data")
        print("6. Exit")
        print("="*50)
    
    def run(self):
        """메인 실행 함수"""
        while True:
            self.show_menu()
            try:
                choice = input("Select an option (1-6): ").strip()
                if choice == "1":
                    username = input("Enter user name to register: ").strip()
                    if username:
                        self.register_face_with_camera(username)
                    else:
                        print("[X] Please enter a name.")
                elif choice == "2":
                    self.realtime_face_recognition()
                elif choice == "3":
                    self.detailed_face_analysis()
                elif choice == "4":
                    self.show_registered_users()
                elif choice == "5":
                    self.delete_user()
                elif choice == "6":
                    print("[EXIT] Program terminated.")
                    break
                else:
                    print("[X] Please enter a number between 1 and 6.")
            except KeyboardInterrupt:
                print("\n[EXIT] Program terminated.")
                break
            except Exception as e:
                print(f"[ERROR] {e}")

def main():
    print("🎯 Unified Face Recognition Program started...")
    face_unified = FaceUnified()
    face_unified.run()

if __name__ == "__main__":
    main() 