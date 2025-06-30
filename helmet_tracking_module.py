import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class HelmetTracker:
    """
    A class for tracking people and detecting helmet usage using YOLOv5 and DeepSORT.
    
    This module combines object detection (YOLOv5) with object tracking (DeepSORT)
    to monitor helmet usage in real-time video streams.
    """
    
    def __init__(self, model_path='/home/lws/kulws2025/kubig2025/final_project/yolov5/helmet_detection/helmet_detection/weights/best.pt', 
                 conf_thresh=0.2, max_age=30, device='auto'):
        """
        Initialize the HelmetTracker.
        
        Args:
            model_path (str): Path to the YOLOv5 custom model weights
            conf_thresh (float): Confidence threshold for detections (0.0-1.0)
            max_age (int): Maximum age for track persistence in DeepSORT
            device (str): Device to run inference on ('auto', 'cuda', 'cpu')
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

        # 트래킹 ID와 헬멧 착용 상태 매핑
        self.id_has_helmet = {}
        
        # Statistics
        self.frame_count = 0
        self.total_detections = 0

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
        Process a single frame for helmet detection and tracking.
        
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

        # DeepSORT 트래킹 업데이트
        tracks = self.tracker.update_tracks(person_dets, frame=frame)

        # 통계 정보
        stats = {
            'persons_detected': len(person_dets),
            'helmets_detected': len(helmet_boxes),
            'no_helmets_detected': len(no_helmet_boxes),
            'tracks_active': len([t for t in tracks if t.is_confirmed()]),
            'people_with_helmets': 0,
            'people_without_helmets': 0
        }

        # 각 트랙별로 상태 업데이트 및 시각화
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            track_id = track.track_id
            
            # 머리 영역 기준 (상단 30%)
            head_box = [x1, y1, x2, y1 + int((y2 - y1) * 0.3)]

            # IoU로 헬멧 착용 여부 판단
            has_helmet = any(self.compute_iou(head_box, helmet) > iou_threshold for helmet in helmet_boxes)
            self.id_has_helmet[track_id] = has_helmet

            # 통계 업데이트
            if has_helmet:
                stats['people_with_helmets'] += 1
            else:
                stats['people_without_helmets'] += 1

            if draw_detections:
                # 박스 색상 설정
                color = (0, 255, 0) if has_helmet else (0, 0, 255)  # 초록 or 빨강
                status = "Helmet" if has_helmet else "No Helmet"

                # 시각화
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id} {status}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.total_detections += stats['persons_detected']
        
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
            'active_tracks': len(self.id_has_helmet),
            'people_with_helmets': sum(1 for has_helmet in self.id_has_helmet.values() if has_helmet),
            'people_without_helmets': sum(1 for has_helmet in self.id_has_helmet.values() if not has_helmet)
        }

    def reset_tracker(self):
        """Reset the tracker state."""
        self.tracker = DeepSort(max_age=30, n_init=3)
        self.id_has_helmet = {}
        self.frame_count = 0
        self.total_detections = 0
