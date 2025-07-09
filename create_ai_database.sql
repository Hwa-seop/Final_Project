-- AI Safety Monitoring Database 생성
CREATE DATABASE IF NOT EXISTS AI_database
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE AI_database;

-- 1. 트래킹 기록 테이블 (tracking_records)
CREATE TABLE IF NOT EXISTS tracking_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    track_id INT NOT NULL,
    helmet_status ENUM('helmet_wearing', 'no_helmet', 'unknown') NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    x1 INT NOT NULL,
    y1 INT NOT NULL,
    x2 INT NOT NULL,
    y2 INT NOT NULL,
    frame_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- 인덱스 추가
    INDEX idx_track_id (track_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_helmet_status (helmet_status),
    INDEX idx_frame_id (frame_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. 이벤트 로그 테이블 (event_logs)
CREATE TABLE IF NOT EXISTS event_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    track_id INT NOT NULL,
    event_type ENUM(
        'NO_HELMET_ALERT',
        'DANGER_ZONE_ENTRY',
        'DANGER_ZONE_EXIT',
        'START_TRACKING',
        'STOP_TRACKING',
        'HELMET_REMOVED',
        'HELMET_PUT_ON',
        'SYSTEM_START',
        'SYSTEM_STOP',
        'ERROR_OCCURRED'
    ) NOT NULL,
    event_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    helmet_status ENUM('helmet_wearing', 'no_helmet', 'unknown'),
    description TEXT,
    severity ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL') DEFAULT 'MEDIUM',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 인덱스 추가
    INDEX idx_track_id (track_id),
    INDEX idx_event_type (event_type),
    INDEX idx_event_time (event_time),
    INDEX idx_severity (severity)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. 추가: 위험구역 설정 테이블 (danger_zones)
CREATE TABLE IF NOT EXISTS danger_zones (
    id INT AUTO_INCREMENT PRIMARY KEY,
    zone_name VARCHAR(100) NOT NULL,
    points JSON NOT NULL, -- 위험구역 점들의 좌표 배열
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. 추가: 세션 관리 테이블 (monitoring_sessions)
CREATE TABLE IF NOT EXISTS monitoring_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_name VARCHAR(100) NOT NULL,
    start_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME NULL,
    total_frames INT DEFAULT 0,
    total_detections INT DEFAULT 0,
    total_events INT DEFAULT 0,
    status ENUM('ACTIVE', 'PAUSED', 'STOPPED') DEFAULT 'ACTIVE',
    config JSON, -- 설정 정보 (JSON 형태로 저장)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_status (status),
    INDEX idx_start_time (start_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 5. 뷰 생성: 실시간 통계 뷰
CREATE OR REPLACE VIEW realtime_stats AS
SELECT 
    COUNT(DISTINCT tr.track_id) as active_tracks,
    COUNT(CASE WHEN tr.helmet_status = 'helmet_wearing' THEN 1 END) as helmet_wearing,
    COUNT(CASE WHEN tr.helmet_status = 'no_helmet' THEN 1 END) as no_helmet,
    COUNT(CASE WHEN tr.helmet_status = 'unknown' THEN 1 END) as unknown_status,
    AVG(tr.confidence) as avg_confidence,
    MAX(tr.timestamp) as last_update
FROM tracking_records tr
WHERE tr.timestamp >= DATE_SUB(NOW(), INTERVAL 1 MINUTE);

-- 6. 뷰 생성: 위험 이벤트 요약 뷰
CREATE OR REPLACE VIEW danger_events_summary AS
SELECT 
    DATE(event_time) as event_date,
    event_type,
    COUNT(*) as event_count,
    COUNT(DISTINCT track_id) as affected_tracks
FROM event_logs
WHERE event_type IN ('NO_HELMET_ALERT', 'DANGER_ZONE_ENTRY')
    AND event_time >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
GROUP BY DATE(event_time), event_type
ORDER BY event_date DESC, event_count DESC;

-- 7. 샘플 데이터 삽입 (테스트용)
INSERT INTO monitoring_sessions (session_name, status) VALUES 
('Test Session 1', 'ACTIVE'),
('Test Session 2', 'STOPPED');

-- 8. 트리거 생성: 이벤트 로그 자동 정리 (30일 이상된 데이터)
DELIMITER //
CREATE EVENT IF NOT EXISTS cleanup_old_events
ON SCHEDULE EVERY 1 DAY
DO
BEGIN
    DELETE FROM event_logs 
    WHERE event_time < DATE_SUB(NOW(), INTERVAL 30 DAY);
    
    DELETE FROM tracking_records 
    WHERE timestamp < DATE_SUB(NOW(), INTERVAL 30 DAY);
END//
DELIMITER ;

-- 9. 권한 설정 (필요시)
-- GRANT ALL PRIVILEGES ON AI_database.* TO 'your_username'@'localhost';
-- FLUSH PRIVILEGES;

-- 10. 테이블 생성 확인
SHOW TABLES;
DESCRIBE tracking_records;
DESCRIBE event_logs;
DESCRIBE danger_zones;
DESCRIBE monitoring_sessions; 