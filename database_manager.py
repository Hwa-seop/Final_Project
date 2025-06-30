#!/usr/bin/env python3
"""
Database Manager for AI Safety Monitoring System (MySQL)

This module handles database connections and operations for storing
tracking statistics, events, and system logs.
"""

import mysql.connector
import mysql.connector.pooling
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os

class DatabaseManager:
    """Database manager for AI safety monitoring system."""
    
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
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            self.logger.info("Database connection established successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")
    
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
        
        try:
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
                return True
                
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
        
        try:
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
        
        try:
            with self.connection.cursor() as cursor:
                details_json = json.dumps(details) if details else None
                
                cursor.execute("""
                    INSERT INTO system_logs (session_id, log_level, message, details)
                    VALUES (%s, %s, %s, %s)
                """, (self.current_session_id, log_level, message, details_json))
                
                self.connection.commit()
                return True
                
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
    
    def get_recent_danger_events(self, hours: int = 24) -> List[Dict]:
        """
        Get recent danger events.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of danger events
        """
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                cursor.execute("""
                    SELECT * FROM danger_events 
                    WHERE event_time >= DATE_SUB(NOW(), INTERVAL %s HOUR)
                    ORDER BY event_time DESC
                """, (hours,))
                
                return cursor.fetchall()
                
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

def init_database(config: Dict[str, str] = None) -> DatabaseManager:
    """Initialize database manager."""
    return DatabaseManager(config)

def get_database_manager() -> DatabaseManager:
    """Get database manager instance."""
    return DatabaseManager() 