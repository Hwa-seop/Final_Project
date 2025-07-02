#!/usr/bin/env python3
"""
설정 관리 모듈

애플리케이션의 모든 설정을 중앙에서 관리합니다.
"""

import os
from typing import Dict, Any

class Config:
    """애플리케이션 설정 클래스"""
    
    # 기본 설정
    DEFAULT_CONFIG = {
        'model_path': 'best.pt',  # YOLO 모델 파일 경로
        'conf_thresh': 0.2,  # 신뢰도 임계값
        'iou_threshold': 0.1,  # IoU 임계값
        'max_age': 30,  # 트랙 최대 유지 시간
        'detection_interval': 5,  # 감지 간격 (프레임 단위)
        'device': 'auto',  # 실행 디바이스 (auto/cuda/cpu)
        'source': 0  # 카메라 소스 (0=기본 카메라)
    }
    
    # 데이터베이스 설정
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),  # 데이터베이스 호스트
        'port': os.getenv('DB_PORT', '3306'),  # 데이터베이스 포트
        'database': os.getenv('DB_NAME', 'ai_safety_monitor'),  # 데이터베이스 이름
        'user': os.getenv('DB_USER', 'root'),  # 사용자명
        'password': os.getenv('DB_PASSWORD', ''),  # 비밀번호
        'charset': 'utf8mb4',  # 문자 인코딩
        'autocommit': True  # 자동 커밋
    }
    
    # Flask 설정
    FLASK_CONFIG = {
        'host': '0.0.0.0',
        'port': 5000,
        'debug': False,
        'threaded': True
    }
    
    # 카메라 설정
    CAMERA_CONFIG = {
        'fps': 30,  # 목표 FPS
        'frame_interval': 0.03,  # 프레임 간격 (초)
        'save_interval': 30  # 데이터베이스 저장 간격 (프레임 단위)
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """현재 설정을 반환합니다."""
        return cls.DEFAULT_CONFIG.copy()
    
    @classmethod
    def update_config(cls, new_config: Dict[str, Any]) -> bool:
        """설정을 업데이트합니다."""
        try:
            for key, value in new_config.items():
                if key in cls.DEFAULT_CONFIG:
                    cls.DEFAULT_CONFIG[key] = value
            return True
        except Exception:
            return False
    
    @classmethod
    def get_db_config(cls) -> Dict[str, Any]:
        """데이터베이스 설정을 반환합니다."""
        return cls.DB_CONFIG.copy()
    
    @classmethod
    def get_flask_config(cls) -> Dict[str, Any]:
        """Flask 설정을 반환합니다."""
        return cls.FLASK_CONFIG.copy()
    
    @classmethod
    def get_camera_config(cls) -> Dict[str, Any]:
        """카메라 설정을 반환합니다."""
        return cls.CAMERA_CONFIG.copy() 