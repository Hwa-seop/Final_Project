-- AI 안전 관제 시스템 데이터베이스 스키마 (MySQL)
-- 이 스크립트는 MySQL 데이터베이스 테이블을 생성합니다.

-- 데이터베이스 생성 (필요시)
CREATE DATABASE IF NOT EXISTS ai_safety_monitor;
USE ai_safety_monitor;

-- 사용자 테이블 (관리자 정보)
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100),
    role VARCHAR(20) DEFAULT 'admin',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 세션 테이블 (모니터링 세션 정보)
CREATE TABLE IF NOT EXISTS monitoring_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_name VARCHAR(100) NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    status VARCHAR(20) DEFAULT 'active',
    config_json TEXT, -- JSON 형태로 설정 저장
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 실시간 통계 테이블 (프레임별 통계)
CREATE TABLE IF NOT EXISTS frame_statistics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT,
    frame_number INT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    persons_detected INT DEFAULT 0,
    people_with_helmets INT DEFAULT 0,
    people_without_helmets INT DEFAULT 0,
    people_in_danger_zone INT DEFAULT 0,
    tracks_active INT DEFAULT 0,
    detection_confidence FLOAT,
    processing_time_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES monitoring_sessions(id) ON DELETE CASCADE
);

-- 추적 객체 테이블 (개별 추적 ID 정보)
CREATE TABLE IF NOT EXISTS tracked_objects (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT,
    track_id INT NOT NULL,
    first_detected TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_frames INT DEFAULT 1,
    has_helmet BOOLEAN DEFAULT FALSE,
    in_danger_zone BOOLEAN DEFAULT FALSE,
    danger_events_count INT DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES monitoring_sessions(id) ON DELETE CASCADE
);

-- 위험 이벤트 테이블 (위험 상황 기록)
CREATE TABLE IF NOT EXISTS danger_events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT,
    track_id INT NOT NULL,
    event_type VARCHAR(50) NOT NULL, -- 'no_helmet_in_danger_zone', 'helmet_removed_in_danger_zone'
    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    location_x INT,
    location_y INT,
    confidence FLOAT,
    description TEXT,
    severity VARCHAR(20) DEFAULT 'medium', -- 'low', 'medium', 'high', 'critical'
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES monitoring_sessions(id) ON DELETE CASCADE
);

-- ROI 설정 테이블 (위험 구역 정보)
CREATE TABLE IF NOT EXISTS roi_configurations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT,
    roi_name VARCHAR(100),
    points_json TEXT NOT NULL, -- JSON 형태로 좌표 저장
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES monitoring_sessions(id) ON DELETE CASCADE
);

-- 시스템 로그 테이블 (시스템 이벤트 기록)
CREATE TABLE IF NOT EXISTS system_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT,
    log_level VARCHAR(20) NOT NULL, -- 'info', 'warning', 'error', 'critical'
    message TEXT NOT NULL,
    details JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES monitoring_sessions(id) ON DELETE CASCADE
);

-- 인덱스 생성 (성능 최적화)
CREATE INDEX idx_frame_statistics_session_timestamp ON frame_statistics(session_id, timestamp);
CREATE INDEX idx_tracked_objects_session_track ON tracked_objects(session_id, track_id);
CREATE INDEX idx_danger_events_session_time ON danger_events(session_id, event_time);
CREATE INDEX idx_system_logs_session_level ON system_logs(session_id, log_level);

-- 뷰 생성 (통계 조회용)
CREATE OR REPLACE VIEW session_summary AS
SELECT 
    ms.id as session_id,
    ms.session_name,
    ms.start_time,
    ms.end_time,
    ms.status,
    COUNT(DISTINCT fs.id) as total_frames,
    COUNT(DISTINCT tobj.track_id) as unique_objects,
    COUNT(de.id) as total_danger_events,
    AVG(fs.persons_detected) as avg_persons_detected,
    MAX(fs.persons_detected) as max_persons_detected
FROM monitoring_sessions ms
LEFT JOIN frame_statistics fs ON ms.id = fs.session_id
LEFT JOIN tracked_objects tobj ON ms.id = tobj.session_id
LEFT JOIN danger_events de ON ms.id = de.session_id
GROUP BY ms.id, ms.session_name, ms.start_time, ms.end_time, ms.status;

-- 기본 관리자 사용자 생성 (비밀번호: admin123)
INSERT INTO users (username, password_hash, email, role) 
VALUES ('admin', 'pbkdf2:sha256:600000$your_salt_here$hash_here', 'admin@example.com', 'admin')
ON DUPLICATE KEY UPDATE username=username;

-- 트리거 생성 (MySQL용)
DELIMITER //

CREATE TRIGGER update_tracked_objects_updated_at 
    BEFORE UPDATE ON tracked_objects 
    FOR EACH ROW 
BEGIN
    SET NEW.updated_at = CURRENT_TIMESTAMP;
END//

CREATE TRIGGER update_roi_configurations_updated_at 
    BEFORE UPDATE ON roi_configurations 
    FOR EACH ROW 
BEGIN
    SET NEW.updated_at = CURRENT_TIMESTAMP;
END//

DELIMITER ; 