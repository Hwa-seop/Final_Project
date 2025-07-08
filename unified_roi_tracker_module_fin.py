import cv2
import torch
import os
import numpy as np
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

class UnifiedROITracker:
    """
    A unified module for ROI-based tracking with DeepSORT.
    
    This module combines object detection (YOLOv5), DeepSORT tracking, and ROI-based
    danger zone detection. It can detect when tracked objects enter a defined
    region and mark them as dangerous.
    """
    
    def __init__(self, model_path, 
                conf_thresh=0.2, 
                max_age=30, 
                device='auto',
                detection_interval=1,
                threshold=0.5
                ):
        self.threshold = threshold
            
        """
        Initialize the UnifiedROITracker.
        
        Args:
            model_path (str): Path to the YOLOv5 custom model weights
            conf_thresh (float): Confidence threshold for detections (0.0-1.0)
            max_age (int): Maximum age for track persistence in DeepSORT
            device (str): Device to run inference on ('auto', 'cuda', 'cpu')
            detection_interval (int): Run detection every N frames (1 = every frame)
        """
        # Auto-detect device if specified
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # YOLOv5 커스텀 모델 로드
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device)
            self.model.conf = conf_thresh
            print(f"Model loaded successfully on {device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # DeepSORT 초기화
        self.tracker = DeepSort(max_age=max_age, n_init=3)

        from face_unified import FaceUnified  # 클래스 상단에서 import

        self.face_recognizer = FaceUnified()
        self.track_id_to_name = {}                # track_id → 이름 매핑 저장 딕셔너리
        self.face_recognizer.face_manager.load_db() # 얼굴 인식용 인스턴스 생성

        # ROI state
        self.roi_points = []
        self.zone_locked = False
        self.zone_poly = None
        
        self.threshold = threshold  # 얼굴 인식에 사용할 유사도 임계값 저장
        
        # Tracking state
        self.id_has_helmet = {}
        self.id_in_danger_zone = {}
        
        # Performance optimization
        self.detection_interval = detection_interval
        self.last_detections = {
            'person_dets': [],
            'helmet_boxes': [],
            'no_helmet_boxes': []
        }
        
        # Statistics
        self.frame_count = 0
        self.total_detections = 0
        self.danger_count = 0

    def setup_roi(self, cap, window_name="ROI Setup"):
        """
        Interactive ROI setup using mouse clicks.
        
        Args:
            cap: VideoCapture object
            window_name (str): Name of the ROI setup window
            
        Returns:
            bool: True if ROI was successfully set up, False otherwise
        """
        print("[INFO] Draw ROI polygon with mouse:")
        print("- Left click: Add point")
        print("- Right click: Remove last point")
        print("- Enter/Space: Confirm ROI")
        print("- q: Quit")
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        while not self.zone_locked:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame for ROI setup")
                return False
            
            output = frame.copy()

            # Draw current ROI points
            for pt in self.roi_points:
                cv2.circle(output, pt, 5, (255, 0, 0), -1)
            
            # Draw ROI lines
            if len(self.roi_points) > 1:
                cv2.polylines(output, [np.array(self.roi_points, dtype=np.int32).reshape((-1,1,2))],
                              isClosed=False, color=(0, 255, 255), thickness=2)

            # Instructions
            cv2.putText(output, "Enter/Space: confirm ROI, q: quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.imshow(window_name, output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow(window_name)
                return False
            elif (key == 13 or key == 32) and len(self.roi_points) >= 3:
                self.zone_poly = np.array(self.roi_points, dtype=np.int32).reshape((-1,1,2))
                self.zone_locked = True

        cv2.destroyWindow(window_name)
        print(f"ROI set with {len(self.roi_points)} points")
        return True

    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for ROI setup."""
        if not self.zone_locked:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and self.roi_points:
                self.roi_points.pop()

    def compute_iou(self, box1, box2):
        """
        Compute Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1 (list): [x1, y1, x2, y2] format
            box2 (list): [x1, y1, x2, y2] format
            
        Returns:
            float: IoU value between 0 and 1
        """
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        
        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter_area
        
        return inter_area / union if union != 0 else 0

    def process_frame(self, frame, draw_detections=True, iou_threshold=0.1):
        """
        Process a single frame for object detection, tracking, and ROI analysis.
        
        Args:
            frame (numpy.ndarray): Input frame (BGR format)
            draw_detections (bool): Whether to draw bounding boxes and labels
            iou_threshold (float): IoU threshold for helmet detection
            
        Returns:
            numpy.ndarray: Processed frame with detections drawn (if enabled)
            dict: Detection statistics
        """
        if frame is None:
            return None, {}
            
        self.frame_count += 1
        
        # Run detection only every N frames for performance
        if self.frame_count % self.detection_interval == 0:
            # YOLO 추론
            results = self.model(frame)
            detections = results.xyxy[0].cpu().numpy()

            person_dets = []
            helmet_boxes = []
            no_helmet_boxes = []

            # 분류별로 검출 결과 정리
            for *xyxy, conf, cls in detections:
                x1, y1, x2, y2 = map(int, xyxy)
                cls = int(cls)
                box = [x1, y1, x2 - x1, y2 - y1]  # x, y, w, h 형식

                if cls == 2:  # person
                    person_dets.append((box, conf, 'person'))
                elif cls == 0:  # helmet
                    helmet_boxes.append([x1, y1, x2, y2])
                elif cls == 1:  # no-helmet
                    no_helmet_boxes.append([x1, y1, x2, y2])
            
            # Store detections for reuse
            self.last_detections = {
                'person_dets': person_dets,
                'helmet_boxes': helmet_boxes,
                'no_helmet_boxes': no_helmet_boxes
            }
        else:
            # Use cached detections
            person_dets = self.last_detections.get('person_dets', [])
            helmet_boxes = self.last_detections.get('helmet_boxes', [])
            no_helmet_boxes = self.last_detections.get('no_helmet_boxes', [])

        # DeepSORT 트래킹 업데이트
        tracks = self.tracker.update_tracks(person_dets, frame=frame)

        # 통계 정보
        stats = {
            'persons_detected': len(person_dets),
            'helmets_detected': len(helmet_boxes),
            'no_helmets_detected': len(no_helmet_boxes),
            'tracks_active': len([t for t in tracks if t.is_confirmed()]),
            'people_with_helmets': 0,
            'people_without_helmets': 0,
            'people_in_danger_zone': 0
        }

        # 각 트랙별로 상태 업데이트 및 시각화
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            track_id = track.track_id
            
             # 얼굴 영역 크롭 후 임시 파일 저장 → 인식
            h = y2 - y1
            face_img = frame[y1:y2, x1:x2]
            try:
                temp_path = f"temp_face_{track_id}.jpg"
                cv2.imwrite(temp_path, face_img)

                found, name, confidence = self.face_recognizer.face_manager.verify_face(temp_path, 
                                                                                        threshold=self.threshold, 
                                                                                        enforce_detection=False)
                # 기존 인식값이 있을 경우 유지, 새로 인식되면 업데이트
                if track_id not in self.track_id_to_name:
                    self.track_id_to_name[track_id] = name if found else "Unknown"
                elif found and self.track_id_to_name[track_id] == "Unknown":
                    self.track_id_to_name[track_id] = name

            except Exception as e:
                print(f"[FaceID 오류] ID {track_id}: {e}")
                if track_id not in self.track_id_to_name:
                    self.track_id_to_name[track_id] = "Error"

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
            # 머리 영역 기준 (상단 20%)
            head_box = [x1, y1, x2, y1 + int((y2 - y1) * 0.2)]

            # IoU로 헬멧 착용 여부 판단
            has_helmet = any(self.compute_iou(head_box, helmet) > iou_threshold for helmet in helmet_boxes)
            self.id_has_helmet[track_id] = has_helmet

            # ROI 위험 구역 체크 (안전한 방식으로)
            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)
            
            in_danger_zone = False
            if self.zone_locked and self.zone_poly is not None:
                try:
                    in_danger_zone = cv2.pointPolygonTest(self.zone_poly, (centroid_x, centroid_y), False) >= 0
                except:
                    in_danger_zone = False
            
            self.id_in_danger_zone[track_id] = in_danger_zone

            # 통계 업데이트
            if has_helmet:
                stats['people_with_helmets'] += 1
            else:
                stats['people_without_helmets'] += 1
            
            if in_danger_zone:
                stats['people_in_danger_zone'] += 1

            if draw_detections:
                # 박스 색상 설정 (위험 구역 우선)
                if in_danger_zone:
                    color = (0, 0, 255)  # 빨강 - 위험 구역
                    status = "DANGER ZONE"
                elif has_helmet:
                    color = (0, 255, 0)  # 초록 - 헬멧 착용
                    status = "Helmet"
                else:
                    color = (0, 165, 255)  # 주황 - 헬멧 미착용
                    status = "No Helmet"

                # 시각화
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                display_name = self.track_id_to_name.get(track_id, "Unknown")
                cv2.putText(frame, f"{display_name} (ID:{track_id}) - {status}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 중심점 표시
                cv2.circle(frame, (centroid_x, centroid_y), 4, (255, 0, 0), -1)

        # ROI 폴리곤 그리기
        if self.zone_locked and self.zone_poly is not None and draw_detections:
            cv2.polylines(frame, [self.zone_poly], isClosed=True, color=(0,0,255), thickness=3)

        self.total_detections += stats['persons_detected']
        self.danger_count += stats['people_in_danger_zone']
        
        return frame, stats

    def get_statistics(self):
        """
        Get tracking statistics.
        
        Returns:
            dict: Statistics about the tracking session
        """
        return {
            'frames_processed': self.frame_count,
            'total_detections': self.total_detections,
            'total_danger_events': self.danger_count,
            'active_tracks': len(self.id_has_helmet),
            'people_with_helmets': sum(1 for has_helmet in self.id_has_helmet.values() if has_helmet),
            'people_without_helmets': sum(1 for has_helmet in self.id_has_helmet.values() if not has_helmet),
            'people_in_danger_zone': sum(1 for in_danger in self.id_in_danger_zone.values() if in_danger),
            'roi_set': self.zone_locked,
            'roi_points': len(self.roi_points) if self.roi_points else 0
        }

    def reset_tracker(self):
        """Reset the tracker state."""
        self.tracker = DeepSort(max_age=30, n_init=3)
        self.id_has_helmet = {}
        self.id_in_danger_zone = {}
        self.frame_count = 0
        self.total_detections = 0
        self.danger_count = 0

    def set_roi(self, points):
        """
        Set ROI programmatically.
        
        Args:
            points (list): List of (x, y) points defining the ROI polygon
        """
        if len(points) >= 3:
            self.roi_points = points
            self.zone_poly = np.array(points, dtype=np.int32).reshape((-1,1,2))
            self.zone_locked = True
            print(f"ROI set with {len(points)} points")
        else:
            print("Error: ROI requires at least 3 points")

    def clear_roi(self):
        """Clear the current ROI."""
        self.roi_points = []
        self.zone_poly = None
        self.zone_locked = False
        self.id_in_danger_zone = {}  # Clear danger zone status
        print("ROI cleared") 