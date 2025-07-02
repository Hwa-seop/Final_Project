#!/usr/bin/env python3
"""
Database Manager for AI Safety Monitoring System (MySQL) - Patched Version

This module handles database connections and operations for storing
tracking statistics, events, and system logs with automatic reconnection.
"""

import mysql.connector
import mysql.connector.pooling
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import os
import time

class DatabaseManager:
    """Database manager for AI safety monitoring system with auto-reconnection."""
    
    def __init__(self, db_config: Dict[str, Any] = None):
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
            'password': os.getenv('DB_PASSWORD', 'qwe123'),
            'charset': 'utf8mb4',
            'autocommit': True,
            'ssl_disabled': True,
            'connection_timeout': 10,
            'pool_size': 5,
            'pool_reset_session': True
        }
        self.connection = None
        self.current_session_id = None
        self.last_connection_check = 0
        self.connection_check_interval = 30  # 30초마다 연결 상태 체크
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def is_connected(self) -> bool:
        """Check if database connection is alive."""
        if not self.connection:
            return False
        
        try:
            # 간단한 쿼리로 연결 상태 확인
            self.connection.ping(reconnect=False, attempts=1, delay=0)
            return True
        except Exception:
            return False
    
    def ensure_connection(self) -> bool:
        """Ensure database connection is alive, reconnect if necessary."""
        current_time = time.time()
        
        # 연결 상태 체크 간격 조절
        if current_time - self.last_connection_check < self.connection_check_interval:
            return self.connection is not None
        
        self.last_connection_check = current_time
        
        if not self.is_connected():
            self.logger.warning("Database connection lost, attempting to reconnect...")
            return self.connect()
        
        return True
    
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            if self.connection:
                try:
                    self.connection.close()
                except:
                    pass
            
            # 연결 설정에서 불필요한 키 제거
            connect_config = {k: v for k, v in self.db_config.items() 
                            if k not in ['pool_size', 'pool_reset_session']}
            
            self.connection = mysql.connector.connect(**connect_config)
            self.logger.info("Database connection established successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            try:
                self.connection.close()
                self.logger.info("Database connection closed")
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")
            finally:
                self.connection = None
    
    def execute_with_retry(self, operation, max_retries=3):
        """Execute database operation with retry logic."""
        for attempt in range(max_retries):
            try:
                if not self.ensure_connection():
                    raise Exception("Failed to establish database connection")
                
                return operation()
                
            except mysql.connector.Error as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Database operation failed (attempt {attempt + 1}): {e}")
                    time.sleep(1)  # 1초 대기 후 재시도
                    self.disconnect()  # 연결 재설정
                else:
                    self.logger.error(f"Database operation failed after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error in database operation: {e}")
                raise
    
    def create_session(self, session_name: str, config: Dict = None) -> Optional[int]:
        """
        Create a new monitoring session.
        
        Args:
            session_name: Name of the monitoring session
            config: Configuration dictionary
            
        Returns:
            Session ID if successful, None otherwise
        """
        def _create_session():
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
        
        try:
            return self.execute_with_retry(_create_session)
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
        def _end_session():
            session_id_to_end = session_id or self.current_session_id
            if not session_id_to_end:
                return False
            
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE monitoring_sessions 
                    SET end_time = CURRENT_TIMESTAMP, status = 'completed'
                    WHERE id = %s
                """, (session_id_to_end,))
                
                self.connection.commit()
                self.logger.info(f"Ended monitoring session: {session_id_to_end}")
                return True
        
        try:
            return self.execute_with_retry(_end_session)
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
        
        def _save_frame_stats():
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
        
        try:
            return self.execute_with_retry(_save_frame_stats)
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
        
        def _update_tracked_object():
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
        
        try:
            return self.execute_with_retry(_update_tracked_object)
        except Exception as e:
            self.logger.error(f"Failed to update tracked object: {e}")
            return False
    
    def record_danger_event(self, track_id: int, event_type: str, 
                          location: tuple = None, confidence: float = 0.0,
                          description: str = None) -> bool:
        """
        Record a danger event.
        
        Args:
            track_id: Tracking ID
            event_type: Type of danger event
            location: (x, y) coordinates
            confidence: Detection confidence
            description: Event description
            
        Returns:
            True if successful, False otherwise
        """
        if not self.current_session_id:
            return False
        
        def _record_danger_event():
            with self.connection.cursor() as cursor:
                location_x, location_y = location if location else (None, None)
                
                cursor.execute("""
                    INSERT INTO danger_events (
                        session_id, track_id, event_type, location_x, location_y,
                        confidence, description
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    self.current_session_id,
                    track_id,
                    event_type,
                    location_x,
                    location_y,
                    confidence,
                    description
                ))
                
                # Update danger events count for tracked object
                cursor.execute("""
                    UPDATE tracked_objects 
                    SET danger_events_count = danger_events_count + 1
                    WHERE session_id = %s AND track_id = %s
                """, (self.current_session_id, track_id))
                
                self.connection.commit()
                self.logger.info(f"Danger event recorded: {event_type} for track {track_id}")
                return True
        
        try:
            return self.execute_with_retry(_record_danger_event)
        except Exception as e:
            self.logger.error(f"Failed to record danger event: {e}")
            return False
    
    def save_roi_configuration(self, roi_name: str, points: List[tuple]) -> bool:
        """
        Save ROI configuration.
        
        Args:
            roi_name: Name of the ROI
            points: List of (x, y) coordinates
            
        Returns:
            True if successful, False otherwise
        """
        if not self.current_session_id:
            return False
        
        def _save_roi_config():
            with self.connection.cursor() as cursor:
                points_json = json.dumps(points)
                
                cursor.execute("""
                    INSERT INTO roi_configurations (session_id, roi_name, points_json)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                    points_json = VALUES(points_json), updated_at = CURRENT_TIMESTAMP
                """, (self.current_session_id, roi_name, points_json))
                
                self.connection.commit()
                return True
        
        try:
            return self.execute_with_retry(_save_roi_config)
        except Exception as e:
            self.logger.error(f"Failed to save ROI configuration: {e}")
            return False
    
    def log_system_event(self, log_level: str, message: str, details: Dict = None) -> bool:
        """
        Log a system event.
        
        Args:
            log_level: Log level ('info', 'warning', 'error', 'critical')
            message: Log message
            details: Additional details as JSON
            
        Returns:
            True if successful, False otherwise
        """
        if not self.current_session_id:
            return False
        
        def _log_system_event():
            with self.connection.cursor() as cursor:
                details_json = json.dumps(details) if details else None
                
                cursor.execute("""
                    INSERT INTO system_logs (session_id, log_level, message, details)
                    VALUES (%s, %s, %s, %s)
                """, (self.current_session_id, log_level, message, details_json))
                
                self.connection.commit()
                return True
        
        try:
            return self.execute_with_retry(_log_system_event)
        except Exception as e:
            self.logger.error(f"Failed to log system event: {e}")
            return False
    
    def get_session_summary(self, session_id: int = None) -> Optional[Dict]:
        """
        Get session summary statistics.
        
        Args:
            session_id: Session ID (uses current session if None)
            
        Returns:
            Summary dictionary or None
        """
        def _get_session_summary():
            session_id_to_use = session_id or self.current_session_id
            if not session_id_to_use:
                return None
            
            with self.connection.cursor(dictionary=True) as cursor:
                cursor.execute("""
                    SELECT * FROM session_summary WHERE session_id = %s
                """, (session_id_to_use,))
                
                return cursor.fetchone()
        
        try:
            return self.execute_with_retry(_get_session_summary)
        except Exception as e:
            self.logger.error(f"Failed to get session summary: {e}")
            return None
    
    def get_recent_danger_events(self, hours: int = 24) -> List[Dict]:
        """
        Get recent danger events.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of danger events
        """
        def _get_recent_events():
            with self.connection.cursor(dictionary=True) as cursor:
                cursor.execute("""
                    SELECT * FROM danger_events 
                    WHERE event_time >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                    ORDER BY event_time DESC
                """, (hours,))
                
                return cursor.fetchall()
        
        try:
            return self.execute_with_retry(_get_recent_events) or []
        except Exception as e:
            self.logger.error(f"Failed to get recent danger events: {e}")
            return []
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

def init_database(config: Dict[str, Any] = None) -> DatabaseManager:
    """Initialize database manager."""
    return DatabaseManager(config)

def get_database_manager() -> DatabaseManager:
    """Get database manager instance."""
    return DatabaseManager()
