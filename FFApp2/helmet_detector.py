#!/usr/bin/env python3
"""
헬멧 감지 및 추적 모듈

YOLOv5 + DeepSORT 기반으로 사람/헬멧/미착용자 감지 및 추적을 수행합니다.
"""

import cv2
import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from deep_sort_realtime.deepsort_tracker import DeepSort

class HelmetDetector:
    """
    YOLOv5 기반 헬멧 감지 및 DeepSORT 추적 클래스
    """
    
    def __init__(self, model_path: str = 'best.pt', conf_thresh: float = 0.2, 
                 max_age: int = 30, device: str = 'auto', detection_interval: int = 1):
        """
        헬멧 감지기 초기화
        
        Args:
            model_path: YOLO 모델 파일 경로
            conf_thresh: 신뢰도 임계값
            max_age: 트랙 최대 유지 시간
            device: 실행 디바이스 (auto/cuda/cpu)
            detection_interval: 감지 간격 (프레임 단위)
        """
        # 디바이스 자동 감지
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # YOLOv5 모델 로드
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device)
            self.model.conf = conf_thresh
            print(f"YOLO 모델 로드 완료: {device}")
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            raise

        # DeepSORT 트래커 초기화
        self.tracker = DeepSort(max_age=max_age, n_init=3)

        # ROI 상태
        self.roi_points = []
        self.zone_locked = False
        self.zone_poly = None
        
        # 추적 상태
        self.id_has_helmet = {}
        self.id_in_danger_zone = {}
        
        # 성능 최적화
        self.detection_interval = detection_interval
        self.last_detections = []
        
        # 통계
        self.frame_count = 0
        self.total_detections = 0
        self.danger_count = 0

    def compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        두 바운딩 박스 간의 IoU를 계산합니다.
        
        Args:
            box1: [x1, y1, x2, y2] 형식
            box2: [x1, y1, x2, y2] 형식
            
        Returns:
            float: IoU 값 (0-1)
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

    def process_frame(self, frame: np.ndarray, draw_detections: bool = True, 
                     iou_threshold: float = 0.1) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        프레임을 처리하여 객체 감지, 추적, ROI 분석을 수행합니다.
        
        Args:
            frame: 입력 프레임 (BGR 형식)
            draw_detections: 바운딩 박스 및 라벨 그리기 여부
            iou_threshold: 헬멧 감지를 위한 IoU 임계값
            
        Returns:
            Tuple[np.ndarray, Dict]: 처리된 프레임과 통계 정보
        """
        if frame is None:
            return None, {}
            
        self.frame_count += 1
        
        # 성능을 위해 N프레임마다 감지 수행
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
            
            # 감지 결과 캐시
            self.last_detections = {
                'person_dets': person_dets,
                'helmet_boxes': helmet_boxes,
                'no_helmet_boxes': no_helmet_boxes
            }
        else:
            # 캐시된 감지 결과 사용
            person_dets = self.last_detections.get('person_dets', [])
            helmet_boxes = self.last_detections.get('helmet_boxes', [])
            no_helmet_boxes = self.last_detections.get('no_helmet_boxes', [])

        # DeepSORT 추적 업데이트
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
            
            # 머리 영역 기준 (상단 20%)
            head_box = [x1, y1, x2, y1 + int((y2 - y1) * 0.2)]

            # IoU로 헬멧 착용 여부 판단
            has_helmet = any(self.compute_iou(head_box, helmet) > iou_threshold for helmet in helmet_boxes)
            self.id_has_helmet[track_id] = has_helmet

            # ROI 위험 구역 체크
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
                cv2.putText(frame, f"ID:{track_id} {status}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 중심점 표시
                cv2.circle(frame, (centroid_x, centroid_y), 4, (255, 0, 0), -1)

        # ROI 폴리곤 그리기
        if self.zone_locked and self.zone_poly is not None and draw_detections:
            cv2.polylines(frame, [self.zone_poly], isClosed=True, color=(0,0,255), thickness=3)

        self.total_detections += stats['persons_detected']
        self.danger_count += stats['people_in_danger_zone']
        
        return frame, stats

    def set_roi(self, points: List[Tuple[int, int]]):
        """
        ROI를 프로그래밍 방식으로 설정합니다.
        
        Args:
            points: (x, y) 좌표 리스트
        """
        if len(points) >= 3:
            self.roi_points = points
            self.zone_poly = np.array(points, dtype=np.int32).reshape((-1,1,2))
            self.zone_locked = True
            print(f"ROI 설정 완료: {len(points)}개 점")
        else:
            print("오류: ROI는 최소 3개의 점이 필요합니다")

    def clear_roi(self):
        """현재 ROI를 초기화합니다."""
        self.roi_points = []
        self.zone_poly = None
        self.zone_locked = False
        self.id_in_danger_zone = {}  # 위험 구역 상태 초기화
        print("ROI 초기화 완료")

    def reset_tracker(self):
        """트래커 상태를 리셋합니다."""
        self.tracker = DeepSort(max_age=30, n_init=3)
        self.id_has_helmet = {}
        self.id_in_danger_zone = {}
        self.frame_count = 0
        self.total_detections = 0
        self.danger_count = 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        추적 통계를 반환합니다.
        
        Returns:
            Dict: 통계 정보
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

    def get_tracker_state(self) -> Dict[str, Any]:
        """
        트래커 상태를 반환합니다.
        
        Returns:
            Dict: 트래커 상태 정보
        """
        return {
            'id_has_helmet': self.id_has_helmet,
            'id_in_danger_zone': self.id_in_danger_zone
        } 