#!/usr/bin/env python3
"""
데이터베이스 관리 모듈

MySQL 데이터베이스 연결 및 데이터 저장을 담당합니다.
"""

import mysql.connector
import mysql.connector.pooling
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
from config import Config

class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    def __init__(self, db_config: Dict[str, str] = None):
        """
        Initialize database manager.
        
        Args:
            db_config: Database configuration dictionary
        """
        self.db_config = db_config or {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '3306')),
            'database': os.getenv('DB_NAME', 'ai_safety_monitor'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', '0000'),
            'charset': 'utf8mb4',
            'autocommit': True
        }
        self.connection = None
        self.current_session_id = None
        self.cursor = None
        self.is_connected = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """
        데이터베이스에 연결합니다.
        
        Returns:
            bool: 연결 성공 여부
        """
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            self.is_connected = True
            
            # 데이터베이스 및 테이블 생성
            self._create_database()
            self._create_tables()
            
            self.logger.info("데이터베이스 연결 성공")
            return True
            
        except mysql.connector.Error as e:
            self.logger.error(f"데이터베이스 연결 오류: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """데이터베이스 연결을 해제합니다."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        self.is_connected = False
        self.logger.info("데이터베이스 연결 해제")
    
    def _create_database(self):
        """데이터베이스를 생성합니다."""
        try:
            # 데이터베이스 생성
            create_db_query = f"CREATE DATABASE IF NOT EXISTS {self.db_config['database']}"
            self.cursor.execute(create_db_query)
            self.connection.commit()
            
            # 데이터베이스 선택
            self.cursor.execute(f"USE {self.db_config['database']}")
            
        except mysql.connector.Error as e:
            self.logger.error(f"데이터베이스 생성 오류: {e}")
    
    def _create_tables(self):
        """필요한 테이블들을 생성합니다."""
        try:
            # 위험 이벤트 테이블
            danger_events_table = """
            CREATE TABLE IF NOT EXISTS danger_events (
                id INT AUTO_INCREMENT PRIMARY KEY,
                event_type VARCHAR(50) NOT NULL,
                track_id INT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                location_x INT,
                location_y INT,
                description TEXT,
                severity_level INT DEFAULT 1,
                resolved BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            # ROI 설정 테이블
            roi_config_table = """
            CREATE TABLE IF NOT EXISTS roi_configurations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                points JSON NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """
            
            # 시스템 로그 테이블
            system_logs_table = """
            CREATE TABLE IF NOT EXISTS system_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                log_level VARCHAR(20) NOT NULL,
                message TEXT NOT NULL,
                module VARCHAR(50),
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            # 통계 테이블
            statistics_table = """
            CREATE TABLE IF NOT EXISTS statistics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                frame_count INT DEFAULT 0,
                persons_detected INT DEFAULT 0,
                helmets_detected INT DEFAULT 0,
                no_helmets_detected INT DEFAULT 0,
                danger_events_count INT DEFAULT 0,
                active_tracks INT DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            # 테이블 생성 실행
            self.cursor.execute(danger_events_table)
            self.cursor.execute(roi_config_table)
            self.cursor.execute(system_logs_table)
            self.cursor.execute(statistics_table)
            
            self.connection.commit()
            self.logger.info("테이블 생성 완료")
            
        except mysql.connector.Error as e:
            self.logger.error(f"테이블 생성 오류: {e}")
    
    def create_session(self, session_name: str, config: Dict = None) -> Optional[int]:
        """
        Create a new monitoring session.
        
        Args:
            session_name: Name of the monitoring session
            config: Configuration dictionary
            
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            with self.connection.cursor() as cursor:
                config_json = json.dumps(config) if config else None
                
                cursor.execute("""
                    INSERT INTO monitoring_sessions (session_name, config_json, status)
                    VALUES (%s, %s, 'active')
                """, (session_name, config_json))
                
                session_id = cursor.lastrowid
                self.connection.commit()
                
                self.current_session_id = session_id
                self.logger.info(f"Created monitoring session: {session_name} (ID: {session_id})")
                return session_id
                
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            return None
    
    def end_session(self, session_id: int = None) -> bool:
        """
        End a monitoring session.
        
        Args:
            session_id: Session ID to end (uses current session if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session_id = session_id or self.current_session_id
            if not session_id:
                return False
            
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE monitoring_sessions 
                    SET end_time = CURRENT_TIMESTAMP, status = 'completed'
                    WHERE id = %s
                """, (session_id,))
                
                self.connection.commit()
                self.logger.info(f"Ended monitoring session: {session_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to end session: {e}")
            return False
    
    def save_frame_statistics(self, stats: Dict[str, Any], frame_number: int) -> bool:
        """
        Save frame statistics to database.
        
        Args:
            stats: Statistics dictionary
            frame_number: Current frame number
            
        Returns:
            True if successful, False otherwise
        """
        if not self.current_session_id:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO frame_statistics (
                        session_id, frame_number, persons_detected, 
                        people_with_helmets, people_without_helmets,
                        people_in_danger_zone, tracks_active, detection_confidence
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    self.current_session_id,
                    frame_number,
                    stats.get('persons_detected', 0),
                    stats.get('people_with_helmets', 0),
                    stats.get('people_without_helmets', 0),
                    stats.get('people_in_danger_zone', 0),
                    stats.get('tracks_active', 0),
                    stats.get('detection_confidence', 0.0)
                ))
                
                self.connection.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save frame statistics: {e}")
            return False
    
    def update_tracked_object(self, track_id: int, has_helmet: bool, 
                            in_danger_zone: bool, location: tuple = None) -> bool:
        """
        Update tracked object information.
        
        Args:
            track_id: Tracking ID
            has_helmet: Whether the person has a helmet
            in_danger_zone: Whether the person is in danger zone
            location: (x, y) coordinates
            
        Returns:
            True if successful, False otherwise
        """
        if not self.current_session_id:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                # Check if object exists
                cursor.execute("""
                    SELECT id FROM tracked_objects 
                    WHERE session_id = %s AND track_id = %s
                """, (self.current_session_id, track_id))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing object
                    cursor.execute("""
                        UPDATE tracked_objects 
                        SET last_seen = CURRENT_TIMESTAMP, total_frames = total_frames + 1,
                            has_helmet = %s, in_danger_zone = %s
                        WHERE session_id = %s AND track_id = %s
                    """, (has_helmet, in_danger_zone, self.current_session_id, track_id))
                else:
                    # Create new object
                    cursor.execute("""
                        INSERT INTO tracked_objects (
                            session_id, track_id, has_helmet, in_danger_zone
                        ) VALUES (%s, %s, %s, %s)
                    """, (self.current_session_id, track_id, has_helmet, in_danger_zone))
                
                self.connection.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update tracked object: {e}")
            return False
    
    def record_danger_event(self, event_type: str, track_id: Optional[int] = None, 
                          location_x: Optional[int] = None, location_y: Optional[int] = None,
                          description: str = "", severity_level: int = 1) -> bool:
        """
        위험 이벤트를 데이터베이스에 기록합니다.
        
        Args:
            event_type: 이벤트 타입 (예: 'no_helmet_in_roi', 'person_in_danger_zone')
            track_id: 트랙 ID
            location_x: X 좌표
            location_y: Y 좌표
            description: 이벤트 설명
            severity_level: 심각도 레벨 (1-5)
            
        Returns:
            bool: 기록 성공 여부
        """
        if not self.is_connected:
            return False
            
        try:
            query = """
            INSERT INTO danger_events 
            (event_type, track_id, location_x, location_y, description, severity_level)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            self.cursor.execute(query, (event_type, track_id, location_x, location_y, 
                                      description, severity_level))
            self.connection.commit()
            return True
            
        except mysql.connector.Error as e:
            self.logger.error(f"위험 이벤트 기록 오류: {e}")
            return False
    
    def save_roi_configuration(self, name: str, points: List[tuple]) -> bool:
        """
        ROI 설정을 데이터베이스에 저장합니다.
        
        Args:
            name: ROI 설정 이름
            points: ROI 점들의 좌표 리스트
            
        Returns:
            bool: 저장 성공 여부
        """
        if not self.is_connected:
            return False
            
        try:
            # 기존 활성 설정을 비활성화
            self.cursor.execute("UPDATE roi_configurations SET is_active = FALSE")
            
            # 새로운 설정 저장
            query = """
            INSERT INTO roi_configurations (name, points, is_active)
            VALUES (%s, %s, TRUE)
            """
            
            points_json = json.dumps(points)
            self.cursor.execute(query, (name, points_json))
            self.connection.commit()
            return True
            
        except mysql.connector.Error as e:
            self.logger.error(f"ROI 설정 저장 오류: {e}")
            return False
    
    def get_active_roi_configuration(self) -> Optional[Dict[str, Any]]:
        """
        현재 활성화된 ROI 설정을 반환합니다.
        
        Returns:
            Dict: ROI 설정 정보 또는 None
        """
        if not self.is_connected:
            return None
            
        try:
            query = "SELECT * FROM roi_configurations WHERE is_active = TRUE ORDER BY id DESC LIMIT 1"
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            
            if result:
                return {
                    'id': result[0],
                    'name': result[1],
                    'points': json.loads(result[2]),
                    'is_active': result[3],
                    'created_at': result[4],
                    'updated_at': result[5]
                }
            return None
            
        except mysql.connector.Error as e:
            self.logger.error(f"ROI 설정 조회 오류: {e}")
            return None
    
    def log_system_event(self, log_level: str, message: str, module: str = "") -> bool:
        """
        시스템 로그를 데이터베이스에 기록합니다.
        
        Args:
            log_level: 로그 레벨 (INFO, WARNING, ERROR, DEBUG)
            message: 로그 메시지
            module: 모듈명
            
        Returns:
            bool: 기록 성공 여부
        """
        if not self.is_connected:
            return False
            
        try:
            query = """
            INSERT INTO system_logs (log_level, message, module)
            VALUES (%s, %s, %s)
            """
            
            self.cursor.execute(query, (log_level, message, module))
            self.connection.commit()
            return True
            
        except mysql.connector.Error as e:
            self.logger.error(f"시스템 로그 기록 오류: {e}")
            return False
    
    def save_statistics(self, stats: Dict[str, Any]) -> bool:
        """
        통계 정보를 데이터베이스에 저장합니다.
        
        Args:
            stats: 통계 정보 딕셔너리
            
        Returns:
            bool: 저장 성공 여부
        """
        if not self.is_connected:
            return False
            
        try:
            query = """
            INSERT INTO statistics 
            (frame_count, persons_detected, helmets_detected, no_helmets_detected, 
             danger_events_count, active_tracks)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            self.cursor.execute(query, (
                stats.get('frames_processed', 0),
                stats.get('people_with_helmets', 0) + stats.get('people_without_helmets', 0),
                stats.get('people_with_helmets', 0),
                stats.get('people_without_helmets', 0),
                stats.get('total_danger_events', 0),
                stats.get('active_tracks', 0)
            ))
            
            self.connection.commit()
            return True
            
        except mysql.connector.Error as e:
            self.logger.error(f"통계 저장 오류: {e}")
            return False
    
    def get_recent_danger_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        최근 위험 이벤트들을 조회합니다.
        
        Args:
            limit: 조회할 이벤트 수
            
        Returns:
            List: 위험 이벤트 리스트
        """
        if not self.is_connected:
            return []
            
        try:
            query = """
            SELECT * FROM danger_events 
            ORDER BY timestamp DESC 
            LIMIT %s
            """
            
            self.cursor.execute(query, (limit,))
            results = self.cursor.fetchall()
            
            events = []
            for result in results:
                events.append({
                    'id': result[0],
                    'event_type': result[1],
                    'track_id': result[2],
                    'timestamp': result[3],
                    'location_x': result[4],
                    'location_y': result[5],
                    'description': result[6],
                    'severity_level': result[7],
                    'resolved': result[8],
                    'created_at': result[9]
                })
            
            return events
            
        except mysql.connector.Error as e:
            self.logger.error(f"위험 이벤트 조회 오류: {e}")
            return []
    
    def get_statistics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        지정된 시간 범위의 통계 요약을 반환합니다.
        
        Args:
            hours: 조회할 시간 범위 (시간 단위)
            
        Returns:
            Dict: 통계 요약 정보
        """
        if not self.is_connected:
            return {}
            
        try:
            query = """
            SELECT 
                COUNT(*) as total_events,
                SUM(CASE WHEN event_type = 'no_helmet_in_roi' THEN 1 ELSE 0 END) as no_helmet_events,
                SUM(CASE WHEN event_type = 'person_in_danger_zone' THEN 1 ELSE 0 END) as danger_zone_events,
                MAX(severity_level) as max_severity
            FROM danger_events 
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
            """
            
            self.cursor.execute(query, (hours,))
            result = self.cursor.fetchone()
            
            return {
                'total_events': result[0] or 0,
                'no_helmet_events': result[1] or 0,
                'danger_zone_events': result[2] or 0,
                'max_severity': result[3] or 0,
                'time_range_hours': hours
            }
            
        except mysql.connector.Error as e:
            self.logger.error(f"통계 요약 조회 오류: {e}")
            return {}
    
    def mark_event_resolved(self, event_id: int) -> bool:
        """
        위험 이벤트를 해결됨으로 표시합니다.
        
        Args:
            event_id: 이벤트 ID
            
        Returns:
            bool: 업데이트 성공 여부
        """
        if not self.is_connected:
            return False
            
        try:
            query = "UPDATE danger_events SET resolved = TRUE WHERE id = %s"
            self.cursor.execute(query, (event_id,))
            self.connection.commit()
            return True
            
        except mysql.connector.Error as e:
            self.logger.error(f"이벤트 해결 표시 오류: {e}")
            return False
    
    def get_session_summary(self, session_id: int = None) -> Optional[Dict]:
        """
        Get session summary statistics.
        
        Args:
            session_id: Session ID (uses current session if None)
            
        Returns:
            Summary dictionary or None
        """
        try:
            session_id = session_id or self.current_session_id
            if not session_id:
                return None
            
            with self.connection.cursor(dictionary=True) as cursor:
                cursor.execute("""
                    SELECT * FROM session_summary WHERE session_id = %s
                """, (session_id,))
                
                return cursor.fetchone()
                
        except Exception as e:
            self.logger.error(f"Failed to get session summary: {e}")
            return None

def init_database(config: Dict[str, str] = None) -> DatabaseManager:
    """Initialize database manager."""
    return DatabaseManager(config)

def get_database_manager() -> DatabaseManager:
    """Get database manager instance."""
    return DatabaseManager() 