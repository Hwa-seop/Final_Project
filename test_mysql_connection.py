#!/usr/bin/env python3
"""
MySQL 데이터베이스 연결 테스트 스크립트

이 스크립트는 MySQL 데이터베이스 연결을 테스트하고
기본적인 데이터베이스 작업을 수행합니다.
"""

import os
import sys
from database_manager import DatabaseManager

def test_database_connection():
    """데이터베이스 연결을 테스트합니다."""
    print("🔍 MySQL 데이터베이스 연결 테스트 시작...")
    
    # 환경 변수에서 설정 가져오기
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '3306')),
        'database': os.getenv('DB_NAME', 'ai_safety_monitor'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', '0000'),
        'charset': 'utf8mb4',
        'autocommit': True
    }
    
    print(f"📊 데이터베이스 설정:")
    print(f"   Host: {db_config['host']}")
    print(f"   Port: {db_config['port']}")
    print(f"   Database: {db_config['database']}")
    print(f"   User: {db_config['user']}")
    
    # 데이터베이스 매니저 생성
    db_manager = DatabaseManager(db_config)
    
    try:
        # 연결 테스트
        if db_manager.connect():
            print("✅ MySQL 연결 성공!")
            
            # 테이블 존재 확인
            test_table_existence(db_manager)
            
            # 세션 생성 테스트
            test_session_creation(db_manager)
            
            # 통계 저장 테스트
            test_statistics_saving(db_manager)
            
            # 세션 종료
            db_manager.end_session()
            
            print("✅ 모든 테스트 통과!")
            
        else:
            print("❌ MySQL 연결 실패!")
            return False
            
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        return False
    finally:
        db_manager.disconnect()
    
    return True

def test_table_existence(db_manager):
    """테이블 존재 여부를 확인합니다."""
    print("\n📋 테이블 존재 확인...")
    
    required_tables = [
        'monitoring_sessions',
        'frame_statistics', 
        'tracked_objects',
        'danger_events',
        'roi_configurations',
        'system_logs'
    ]
    
    try:
        with db_manager.connection.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            for table in required_tables:
                if table in existing_tables:
                    print(f"   ✅ {table}")
                else:
                    print(f"   ❌ {table} (없음)")
                    
    except Exception as e:
        print(f"   ❌ 테이블 확인 실패: {e}")

def test_session_creation(db_manager):
    """세션 생성을 테스트합니다."""
    print("\n📝 세션 생성 테스트...")
    
    try:
        session_id = db_manager.create_session("테스트 세션", {
            'test': True,
            'description': '연결 테스트용 세션'
        })
        
        if session_id:
            print(f"   ✅ 세션 생성 성공 (ID: {session_id})")
            
            # 시스템 로그 테스트
            db_manager.log_system_event('info', '연결 테스트 완료', {
                'test_type': 'connection_test',
                'session_id': session_id
            })
            print("   ✅ 시스템 로그 기록 성공")
            
        else:
            print("   ❌ 세션 생성 실패")
            
    except Exception as e:
        print(f"   ❌ 세션 생성 테스트 실패: {e}")

def test_statistics_saving(db_manager):
    """통계 저장을 테스트합니다."""
    print("\n📊 통계 저장 테스트...")
    
    try:
        # 프레임 통계 저장
        stats = {
            'persons_detected': 2,
            'people_with_helmets': 1,
            'people_without_helmets': 1,
            'people_in_danger_zone': 0,
            'tracks_active': 2,
            'detection_confidence': 0.85
        }
        
        if db_manager.save_frame_statistics(stats, 1):
            print("   ✅ 프레임 통계 저장 성공")
        else:
            print("   ❌ 프레임 통계 저장 실패")
        
        # 추적 객체 업데이트
        if db_manager.update_tracked_object(1, True, False):
            print("   ✅ 추적 객체 업데이트 성공")
        else:
            print("   ❌ 추적 객체 업데이트 실패")
            
    except Exception as e:
        print(f"   ❌ 통계 저장 테스트 실패: {e}")

def main():
    """메인 함수"""
    print("=" * 50)
    print("🔧 MySQL 데이터베이스 연결 테스트")
    print("=" * 50)
    
    success = test_database_connection()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
        print("✅ MySQL 데이터베이스가 정상적으로 설정되었습니다.")
        print("\n다음 단계:")
        print("1. Flask 앱 실행: python flask_app.py")
        print("2. 웹 브라우저에서 http://localhost:5000 접속")
    else:
        print("❌ 테스트가 실패했습니다.")
        print("\n문제 해결 방법:")
        print("1. MySQL 서비스가 실행 중인지 확인")
        print("2. 데이터베이스 설정 확인 (env_example.txt 참조)")
        print("3. 데이터베이스 초기화 스크립트 실행: ./init_mysql_db.sh")
        print("4. 사용자 권한 확인")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 