#!/usr/bin/env python3
"""
Configuration file for AI Safety Monitor

이 파일에서 객체 제한, 메모리 관리, 성능 설정을 조정할 수 있습니다.
"""

# === [객체 제한 설정] ===
MAX_OBJECTS = 20  # 최대 추적 객체 수
# 권장값:
# - 저사양 시스템: 10-15개
# - 중간 사양 시스템: 20-30개  
# - 고사양 시스템: 30-50개

# === [메모리 관리 설정] ===
MAX_MEMORY_PERCENT = 80  # 최대 메모리 사용률 (%)
MEMORY_CHECK_INTERVAL = 100  # 메모리 체크 간격 (프레임)
OBJECT_CLEANUP_INTERVAL = 50  # 객체 정리 간격 (프레임)

# === [프레임 처리 설정] ===
MAX_FRAME_SIZE = (640, 480)  # 최대 프레임 크기
JPEG_QUALITY = 70  # JPEG 압축 품질 (1-100)
FRAME_RATE = 30  # 목표 프레임 레이트

# === [YOLO 모델 설정] ===
DEFAULT_CONFIG = {
    'model_path': 'best.pt',
    'conf_thresh': 0.3,  # 신뢰도 임계값 (높을수록 정확하지만 감지율 감소)
    'iou_threshold': 0.1,  # IoU 임계값
    'max_age': 20,  # 추적 지속 시간 (프레임)
    'detection_interval': 3,  # 감지 간격 (프레임)
    'device': 'auto',  # 'auto', 'cuda', 'cpu'
    'source': 0  # 카메라 소스
}

# === [데이터베이스 설정] ===
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'database': 'ai_safety_monitor',
    'user': 'root',
    'password': 'qwe123',
    'charset': 'utf8mb4',
    'autocommit': True
}

# === [성능 최적화 팁] ===
PERFORMANCE_TIPS = {
    "low_end": {
        "max_objects": 10,
        "max_frame_size": (480, 360),
        "jpeg_quality": 60,
        "detection_interval": 5,
        "conf_thresh": 0.4
    },
    "medium": {
        "max_objects": 20,
        "max_frame_size": (640, 480),
        "jpeg_quality": 70,
        "detection_interval": 3,
        "conf_thresh": 0.3
    },
    "high_end": {
        "max_objects": 30,
        "max_frame_size": (800, 600),
        "jpeg_quality": 80,
        "detection_interval": 2,
        "conf_thresh": 0.25
    }
}

def get_performance_config(level="medium"):
    """성능 레벨에 따른 설정을 반환합니다."""
    if level in PERFORMANCE_TIPS:
        config = PERFORMANCE_TIPS[level].copy()
        config.update(DEFAULT_CONFIG)
        return config
    return DEFAULT_CONFIG

def print_system_requirements():
    """시스템 요구사항을 출력합니다."""
    print("=== AI Safety Monitor 시스템 요구사항 ===")
    print(f"📊 최대 객체 수: {MAX_OBJECTS}개")
    print(f"💾 최대 메모리 사용률: {MAX_MEMORY_PERCENT}%")
    print(f"🖼️ 최대 프레임 크기: {MAX_FRAME_SIZE[0]}x{MAX_FRAME_SIZE[1]}")
    print(f"📸 JPEG 품질: {JPEG_QUALITY}%")
    print(f"🎯 목표 프레임 레이트: {FRAME_RATE} FPS")
    print()
    print("=== 성능 레벨별 권장 설정 ===")
    for level, config in PERFORMANCE_TIPS.items():
        print(f"🔧 {level.upper()}:")
        print(f"   - 최대 객체: {config['max_objects']}개")
        print(f"   - 프레임 크기: {config['max_frame_size'][0]}x{config['max_frame_size'][1]}")
        print(f"   - JPEG 품질: {config['jpeg_quality']}%")
        print(f"   - 감지 간격: {config['detection_interval']} 프레임")
        print(f"   - 신뢰도 임계값: {config['conf_thresh']}")
        print()

if __name__ == "__main__":
    print_system_requirements() 