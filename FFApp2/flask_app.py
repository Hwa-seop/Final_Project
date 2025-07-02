#!/usr/bin/env python3
"""
실시간 헬멧 감지 및 ROI 추적 시스템

Flask 기반 웹 애플리케이션으로 YOLOv5 + DeepSORT를 사용하여
헬멧 착용 여부를 감지하고 위험 구역(ROI) 진입을 모니터링합니다.
"""

import os
import sys
import signal
import time
import threading
from typing import Dict, Any

# 모듈 import
from config import Config
from camera_manager import CameraManager
from database_manager import DatabaseManager
from api_routes import create_app

class SafetyMonitoringApp:
    """안전 모니터링 애플리케이션 메인 클래스"""
    
    def __init__(self):
        self.camera_manager = None
        self.db_manager = None
        self.app = None
        self.is_running = False
        
        # 설정 로드
        self.config = Config.get_config()
        self.flask_config = Config.get_flask_config()
        
        # 종료 이벤트
        self.shutdown_event = threading.Event()
        
    def initialize(self) -> bool:
        """
        애플리케이션을 초기화합니다.
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            print("=== AI 안전 모니터링 시스템 초기화 중 ===")
            
            # 데이터베이스 매니저 초기화
            print("1. 데이터베이스 연결 중...")
            self.db_manager = DatabaseManager()
            if not self.db_manager.connect():
                print("오류: 데이터베이스 연결에 실패했습니다.")
                return False
            print("✓ 데이터베이스 연결 성공")
            
            # 카메라 매니저 초기화
            print("2. 카메라 시스템 초기화 중...")
            self.camera_manager = CameraManager()
            if not self.camera_manager.initialize_camera():
                print("경고: 카메라 초기화에 실패했습니다. 웹에서 수동으로 시작할 수 있습니다.")
            else:
                print("✓ 카메라 초기화 성공")
            
            # Flask 애플리케이션 생성
            print("3. 웹 애플리케이션 생성 중...")
            self.app = create_app(self.camera_manager, self.db_manager)
            print("✓ 웹 애플리케이션 생성 완료")
            
            # 시스템 로그 기록
            self.db_manager.log_system_event('INFO', '시스템 초기화 완료', 'main')
            
            print("=== 초기화 완료 ===")
            return True
            
        except Exception as e:
            print(f"초기화 오류: {e}")
            return False
    
    def start(self):
        """애플리케이션을 시작합니다."""
        if not self.initialize():
            print("초기화 실패로 인해 애플리케이션을 시작할 수 없습니다.")
            return
        
        try:
            self.is_running = True
            
            # 카메라 루프 시작
            if self.camera_manager:
                self.camera_manager.start_camera_loop()
                print("카메라 루프 시작됨")
            
            # 통계 저장 스레드 시작
            stats_thread = threading.Thread(target=self._stats_saver_thread)
            stats_thread.daemon = True
            stats_thread.start()
            
            # Flask 서버 시작
            print(f"웹 서버 시작: http://{self.flask_config['host']}:{self.flask_config['port']}")
            self.app.run(
                host=self.flask_config['host'],
                port=self.flask_config['port'],
                debug=self.flask_config['debug'],
                threaded=self.flask_config['threaded']
            )
            
        except KeyboardInterrupt:
            print("\n사용자에 의해 중단됨")
        except Exception as e:
            print(f"애플리케이션 실행 오류: {e}")
        finally:
            self.shutdown()
    
    def _stats_saver_thread(self):
        """통계 저장 스레드"""
        save_interval = Config.get_camera_config()['save_interval']
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                time.sleep(save_interval)
                
                if self.camera_manager and self.db_manager:
                    stats = self.camera_manager.get_detector_stats()
                    if stats:
                        self.db_manager.save_statistics(stats)
                        
            except Exception as e:
                print(f"통계 저장 오류: {e}")
    
    def shutdown(self):
        """애플리케이션을 종료합니다."""
        print("\n=== 시스템 종료 중 ===")
        
        self.is_running = False
        self.shutdown_event.set()
        
        try:
            # 카메라 매니저 종료
            if self.camera_manager:
                self.camera_manager.release()
                print("✓ 카메라 시스템 종료")
            
            # 데이터베이스 연결 종료
            if self.db_manager:
                self.db_manager.log_system_event('INFO', '시스템 종료', 'main')
                self.db_manager.disconnect()
                print("✓ 데이터베이스 연결 종료")
            
            print("=== 시스템 종료 완료 ===")
            
        except Exception as e:
            print(f"종료 중 오류: {e}")

def signal_handler(signum, frame):
    """시그널 핸들러"""
    print(f"\n시그널 {signum} 수신됨. 시스템을 종료합니다.")
    sys.exit(0)

def main():
    """메인 함수"""
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 환경 변수 설정 (필요한 경우)
    os.environ.setdefault('FLASK_ENV', 'production')
    
    # 애플리케이션 시작
    app = SafetyMonitoringApp()
    app.start()

if __name__ == '__main__':
    main() 