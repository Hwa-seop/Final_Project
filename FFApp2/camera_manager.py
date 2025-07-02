#!/usr/bin/env python3
"""
카메라 관리 모듈

카메라 초기화, 프레임 처리, 스트리밍을 담당합니다.
"""

import cv2
import threading
import time
from typing import Optional, Dict, Any
from helmet_detector import HelmetDetector
from config import Config

class CameraManager:
    """카메라 관리 클래스"""
    
    def __init__(self):
        self.camera = None
        self.detector = None
        self.is_running = False
        self.camera_thread = None
        self.output_frame = None
        self.lock = threading.Lock()
        self.frames_processed = 0
        self.last_frame_time = 0
        
        # ROI 관련
        self.roi_drawing_mode = False
        self.roi_points = []
        
        # 설정
        self.config = Config.get_config()
        self.camera_config = Config.get_camera_config()
    
    def initialize_camera(self) -> bool:
        """
        카메라와 헬멧 감지기를 초기화합니다.
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            self.camera = cv2.VideoCapture(self.config['source'])
            if not self.camera.isOpened():
                print("Error: Could not open camera")
                return False
            
            # 헬멧 감지기 초기화
            self.detector = HelmetDetector(
                model_path=self.config['model_path'],
                conf_thresh=self.config['conf_thresh'],
                max_age=self.config['max_age'],
                device=self.config['device'],
                detection_interval=self.config['detection_interval']
            )
            
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def start_camera_loop(self):
        """카메라 처리 루프를 시작합니다."""
        if not self.is_running:
            self.is_running = True
            self.camera_thread = threading.Thread(target=self._camera_loop)
            self.camera_thread.start()
    
    def stop_camera_loop(self):
        """카메라 처리 루프를 중지합니다."""
        self.is_running = False
        if self.camera_thread:
            self.camera_thread.join()
    
    def _camera_loop(self):
        """메인 카메라 처리 루프입니다."""
        while self.is_running:
            if self.camera is None or not self.camera.isOpened():
                time.sleep(0.1)
                continue
            
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            try:
                # 프레임 처리 시간 업데이트
                self.last_frame_time = time.time()
                
                # ROI 그리기 모드일 때 점들 그리기
                if self.roi_drawing_mode and self.roi_points:
                    self._draw_roi_points(frame)
                
                # 헬멧 감지기가 있고 ROI 그리기 모드가 아닐 때 프레임 처리
                if self.detector and not self.roi_drawing_mode:
                    processed_frame, frame_stats = self.detector.process_frame(
                        frame, 
                        draw_detections=True,
                        iou_threshold=self.config['iou_threshold']
                    )
                    self.frames_processed += 1
                else:
                    processed_frame = frame
                    self.frames_processed += 1
                
                if processed_frame is not None:
                    # 웹 스트리밍용으로 프레임 인코딩
                    ret, buffer = cv2.imencode('.jpg', processed_frame)
                    if ret:
                        with self.lock:
                            self.output_frame = buffer.tobytes()
                            
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
            
            time.sleep(self.camera_config['frame_interval'])
    
    def _draw_roi_points(self, frame):
        """ROI 점들을 프레임에 그립니다."""
        for i, point in enumerate(self.roi_points):
            cv2.circle(frame, point, 5, (0, 255, 255), -1)  # 노란색 원으로 점 표시
            cv2.putText(frame, str(i+1), (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # 점 번호 표시
        
        # 점들 사이에 선 그리기
        for i in range(len(self.roi_points)):
            if i < len(self.roi_points) - 1:
                cv2.line(frame, self.roi_points[i], self.roi_points[i+1], (0, 255, 255), 2)
            else:
                cv2.line(frame, self.roi_points[i], self.roi_points[0], (0, 255, 255), 2)
    
    def get_output_frame(self) -> Optional[bytes]:
        """현재 출력 프레임을 반환합니다."""
        with self.lock:
            return self.output_frame
    
    def add_roi_point(self, x: int, y: int) -> bool:
        """ROI 점을 추가합니다."""
        if self.roi_drawing_mode:
            self.roi_points.append((x, y))
            return True
        return False
    
    def finish_roi_drawing(self) -> bool:
        """ROI 그리기를 완료합니다."""
        if len(self.roi_points) >= 3 and self.detector:
            try:
                self.detector.set_roi(self.roi_points)
                self.roi_drawing_mode = False
                return True
            except Exception as e:
                print(f"Error setting ROI: {e}")
                return False
        return False
    
    def clear_roi(self):
        """ROI를 초기화합니다."""
        if self.detector:
            self.detector.clear_roi()
        self.roi_points = []
        self.roi_drawing_mode = False
    
    def start_roi_drawing(self):
        """ROI 그리기 모드를 시작합니다."""
        self.roi_drawing_mode = True
        self.roi_points = []
    
    def cancel_roi_drawing(self):
        """ROI 그리기를 취소합니다."""
        self.roi_drawing_mode = False
        self.roi_points = []
    
    def reset_detector(self):
        """헬멧 감지기를 리셋합니다."""
        if self.detector:
            self.detector.reset_tracker()
        self.frames_processed = 0
    
    def get_detector_stats(self) -> Dict[str, Any]:
        """헬멧 감지기 통계를 반환합니다."""
        if self.detector:
            return self.detector.get_statistics()
        return {}
    
    def get_detector_state(self) -> Dict[str, Any]:
        """헬멧 감지기 상태를 반환합니다."""
        if self.detector:
            return self.detector.get_tracker_state()
        return {}
    
    def update_config(self, new_config: Dict[str, Any]):
        """설정을 업데이트하고 헬멧 감지기를 재초기화합니다."""
        self.config.update(new_config)
        if self.detector:
            self.detector = HelmetDetector(
                model_path=self.config['model_path'],
                conf_thresh=self.config['conf_thresh'],
                max_age=self.config['max_age'],
                device=self.config['device'],
                detection_interval=self.config['detection_interval']
            )
    
    def release(self):
        """카메라 리소스를 해제합니다."""
        self.stop_camera_loop()
        if self.camera:
            self.camera.release()
    
    def get_status(self) -> Dict[str, Any]:
        """카메라 상태를 반환합니다."""
        return {
            'is_running': self.is_running,
            'roi_drawing_mode': self.roi_drawing_mode,
            'roi_points_count': len(self.roi_points),
            'frames_processed': self.frames_processed,
            'last_frame_time': self.last_frame_time,
            'camera_opened': self.camera.isOpened() if self.camera else False
        } 