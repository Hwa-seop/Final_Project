#!/usr/bin/env python3
"""
Face Recognition Functions for Helmet Detection System

This module provides face recognition capabilities for assigning unique IDs
to detected individuals based on their facial features using face_recognition library.
"""

import os
import time
import pickle
import numpy as np
from typing import Dict, Tuple, Optional, List
import cv2

# 얼굴 인식 관련 라이브러리
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ face_recognition 라이브러리 로드 성공")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️ face_recognition 라이브러리가 설치되지 않았습니다.")
    print("설치 방법: pip install face-recognition")

class FaceRecognitionManager:
    """얼굴 인식 관리자 클래스"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.6,
                 max_embeddings: int = 50,
                 face_data_file: str = 'approved_face_encodings.pkl'):
        self.similarity_threshold = similarity_threshold
        self.max_embeddings = max_embeddings
        self.face_data_file = face_data_file
        
        # 얼굴 인식 관련 변수들
        self.face_encodings: Dict[str, np.ndarray] = {}
        self.face_id_counter: int = 0
        self.known_faces: Dict[str, Dict] = {}
        
        # 얼굴 데이터 로드
        self.load_known_faces()
    
    def load_known_faces(self) -> None:
        """저장된 얼굴 데이터를 로드합니다."""
        try:
            if os.path.exists(self.face_data_file):
                with open(self.face_data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('known_faces', {})
                    self.face_encodings = data.get('face_encodings', {})
                    self.face_id_counter = data.get('face_id_counter', 0)
                    print(f"✅ {len(self.known_faces)}개의 알려진 얼굴을 로드했습니다.")
            else:
                print("⚠️ 저장된 얼굴 데이터가 없습니다.")
        except Exception as e:
            print(f"❌ 얼굴 데이터 로드 실패: {e}")
    
    def save_known_faces(self) -> None:
        """얼굴 데이터를 저장합니다."""
        try:
            data = {
                'known_faces': self.known_faces,
                'face_encodings': self.face_encodings,
                'face_id_counter': self.face_id_counter
            }
            with open(self.face_data_file, 'wb') as f:
                pickle.dump(data, f)
            print("✅ 얼굴 데이터가 저장되었습니다.")
        except Exception as e:
            print(f"❌ 얼굴 데이터 저장 실패: {e}")
    
    def extract_face_encoding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 이미지에서 인코딩을 추출합니다."""
        if not FACE_RECOGNITION_AVAILABLE:
            return None
        try:
            # RGB로 변환 (face_recognition은 RGB를 요구함)
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                rgb_img = face_img
            else:
                rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_img)
            if len(encodings) > 0:
                return encodings[0]
            return None
        except Exception as e:
            print(f"얼굴 인코딩 추출 실패: {e}")
            return None
    
    def calculate_face_similarity(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """두 얼굴 인코딩 간의 유사도를 계산합니다."""
        try:
            # face_recognition의 face_distance 함수 사용
            distance = face_recognition.face_distance([encoding1], encoding2)
            # distance는 numpy 배열이므로 첫 번째 요소를 가져옴
            distance_value = float(distance[0])
            # 거리를 유사도로 변환 (거리가 작을수록 유사도가 높음)
            similarity = 1.0 - distance_value
            return similarity
        except Exception as e:
            print(f"유사도 계산 실패: {e}")
            return 0.0
    
    def find_matching_face(self, face_encoding: np.ndarray) -> Tuple[Optional[str], float]:
        """주어진 얼굴 인코딩과 가장 유사한 얼굴을 찾습니다."""
        if face_encoding is None or not self.known_faces:
            return None, 0.0
        
        best_match_id = None
        best_similarity = 0.0
        
        for face_id, stored_encoding in self.face_encodings.items():
            similarity = self.calculate_face_similarity(face_encoding, stored_encoding)
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = face_id
        
        return best_match_id, best_similarity
    
    def register_new_face(self, face_img: np.ndarray, face_encoding: np.ndarray) -> str:
        """새로운 얼굴을 등록합니다."""
        # 최대 얼굴 인코딩 수 제한
        if len(self.face_encodings) >= self.max_embeddings:
            # 가장 오래된 얼굴 제거
            oldest_id = min(self.face_encodings.keys())
            del self.face_encodings[oldest_id]
            if oldest_id in self.known_faces:
                del self.known_faces[oldest_id]
        
        self.face_id_counter += 1
        face_id = f"Person_{self.face_id_counter}"
        
        self.known_faces[face_id] = {
            'id': face_id,
            'first_seen': time.time(),
            'last_seen': time.time(),
            'detection_count': 1
        }
        
        self.face_encodings[face_id] = face_encoding
        
        print(f"🆕 새로운 얼굴 등록: {face_id}")
        return face_id
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> Dict[str, Dict]:
        """프레임에서 얼굴을 감지하고 인식합니다."""
        if not FACE_RECOGNITION_AVAILABLE:
            return {}
        
        try:
            # RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 얼굴 위치 감지
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            detected_faces = {}
            
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                top, right, bottom, left = face_location
                
                # 얼굴 영역 정보
                face_region = {
                    'x': left,
                    'y': top,
                    'w': right - left,
                    'h': bottom - top
                }
                
                # 기존 얼굴과 매칭 시도
                matched_id, similarity = self.find_matching_face(face_encoding)
                
                if matched_id:
                    # 기존 얼굴 업데이트
                    self.known_faces[matched_id]['last_seen'] = time.time()
                    self.known_faces[matched_id]['detection_count'] += 1
                    detected_faces[f"face_{i}"] = {
                        'id': matched_id,
                        'similarity': similarity,
                        'region': face_region,
                        'encoding': face_encoding
                    }
                else:
                    # 새로운 얼굴 등록
                    new_face_id = self.register_new_face(frame, face_encoding)
                    detected_faces[f"face_{i}"] = {
                        'id': new_face_id,
                        'similarity': 1.0,
                        'region': face_region,
                        'encoding': face_encoding
                    }
            
            return detected_faces
            
        except Exception as e:
            print(f"얼굴 감지 실패: {e}")
            return {}
    
    def cleanup_old_faces(self, max_age_hours: int = 1) -> None:
        """오래된 얼굴 데이터를 정리합니다."""
        current_time = time.time()
        faces_to_remove = []
        
        for face_id, face_info in self.known_faces.items():
            if current_time - face_info['last_seen'] > max_age_hours * 3600:
                faces_to_remove.append(face_id)
        
        for face_id in faces_to_remove:
            if face_id in self.known_faces:
                del self.known_faces[face_id]
            if face_id in self.face_encodings:
                del self.face_encodings[face_id]
        
        if faces_to_remove:
            print(f"🧹 {len(faces_to_remove)}개의 오래된 얼굴 데이터를 정리했습니다.")
    
    def get_face_statistics(self) -> Dict:
        """얼굴 인식 통계를 반환합니다."""
        return {
            'total_faces': len(self.known_faces),
            'face_encodings_count': len(self.face_encodings),
            'face_id_counter': self.face_id_counter,
            'known_faces': self.known_faces
        }
    
    def find_nearest_face(self, bbox: List[int], detected_faces: Dict[str, Dict]) -> Optional[str]:
        """바운딩 박스와 가장 가까운 얼굴을 찾습니다."""
        if not detected_faces:
            return None
        
        bbox_center_x = bbox[0] + bbox[2] // 2
        bbox_center_y = bbox[1] + bbox[3] // 2
        
        nearest_face_id = None
        min_distance = float('inf')
        
        for face_key, face_info in detected_faces.items():
            if 'region' in face_info:
                region = face_info['region']
                face_center_x = region['x'] + region['w'] // 2
                face_center_y = region['y'] + region['h'] // 2
                
                distance = ((bbox_center_x - face_center_x) ** 2 + 
                           (bbox_center_y - face_center_y) ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_face_id = face_info.get('id')
        
        return nearest_face_id  