#!/usr/bin/env python3
"""
Flask Web Application for Unified ROI Tracker

This Flask app provides a web interface for real-time helmet detection and ROI tracking.
Users can view live video stream, control settings, and monitor statistics through a web browser.
"""
# 기본 모듈 및 외부 의존성
from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import time
import json
import os
import signal
import sys
from unified_roi_tracker_module import UnifiedROITracker
from database_manager_patched import DatabaseManager, init_database, get_database_manager
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# === [Flask 앱 초기화 및 전역 변수] ===
app = Flask(__name__)

# 시스템 제어 및 상태 관리용 전역 변수들
# Global variables
camera = None  # OpenCV 비디오 소스
tracker = None  # ROI 추적기
output_frame = None  # 현재 출력 프레임
lock = threading.Lock()  # 스레드 안전을 위한 잠금
stats = {}  # 통계 정보
is_running = False  # 트래킹 실행 중 상태
camera_thread = None  # 백그라운드 영상 처리 쓰레드
roi_drawing_mode = False  # ROI 그리기 모드
roi_points = []  # ROI 포인트
frames_processed = 0  # 처리된 프레임 수
last_frame_time = 0  # 마지막 프레임 시간

# ID 기반 추적 및 통계 관리
# Statistics tracking for unique IDs
processed_ids = set()   # 처리된 ID 모음
id_stats = {}  # ID별 통계 정보를 저장하기 위한 딕셔너리
total_danger_events = 0  # 총 위험 이벤트 카운터 (누적)

# Database manager
db_manager = None
current_session_id = None

# === [기본 설정값 정의] ===
DEFAULT_CONFIG = {
    'model_path': 'best.pt',
    'conf_thresh': 0.3,
    'iou_threshold': 0.2,
    'max_age': 30,
    'detection_interval': 5,
    'device': 'auto',
    'source': 0
}
# === [DB 연결 초기화] ===
def initialize_database():
    """Initialize database connection."""
    global db_manager
    
    try:
        # Database configuration (can be set via environment variables)
        db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '3306'),
            'database': os.getenv('DB_NAME', 'ai_safety_monitor'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', 'qwe123'),
            'charset': 'utf8mb4',
            'autocommit': True
        }
        
        db_manager = init_database(db_config)
        if db_manager.connect():
            print("Database connection established")
            return True
        else:
            print("Database connection failed - running without database")
            return False
    except Exception as e:
        print(f"Database initialization error: {e} - running without database")
        return False
# === [세션 생성 (DB 기록용)] ===
def create_monitoring_session():
    """Create a new monitoring session."""
    global db_manager, current_session_id
    
    if not db_manager:
        return None
    
    try:
        session_name = f"AI_Safety_Monitor_{int(time.time())}"
        current_session_id = db_manager.create_session(session_name, DEFAULT_CONFIG)
        
        if current_session_id:
            print(f"Created monitoring session: {session_name} (ID: {current_session_id})")
            db_manager.log_system_event('info', 'Monitoring session started', {
                'session_id': current_session_id,
                'config': DEFAULT_CONFIG
            })
        
        return current_session_id
    except Exception as e:
        print(f"Failed to create monitoring session: {e}")
        return None
# === [카메라 및 트래커 초기화] ===
def initialize_camera():
    """Initialize camera and tracker."""
    global camera, tracker
    
    try:
        camera = cv2.VideoCapture(DEFAULT_CONFIG['source'])
        if not camera.isOpened():
            print("Error: Could not open camera")
            return False
        
        # Initialize tracker
        tracker = UnifiedROITracker(
            model_path=DEFAULT_CONFIG['model_path'],
            conf_thresh=DEFAULT_CONFIG['conf_thresh'],
            max_age=DEFAULT_CONFIG['max_age'],
            device=DEFAULT_CONFIG['device'],
            detection_interval=DEFAULT_CONFIG['detection_interval']
        )
        
        return True
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return False

def update_statistics_for_id(track_id, has_helmet, in_danger_zone):
    """Update statistics for a specific track ID, avoiding duplicates."""
    global processed_ids, id_stats, total_danger_events, db_manager
    
    if track_id not in processed_ids:
        # First time seeing this ID
        processed_ids.add(track_id)
        id_stats[track_id] = {
            'has_helmet': has_helmet,
            'in_danger_zone': in_danger_zone,
            'danger_event_counted': False
        }
        
        # Count danger event only once per ID
        if in_danger_zone and not has_helmet and not id_stats[track_id]['danger_event_counted']:
            total_danger_events += 1
            id_stats[track_id]['danger_event_counted'] = True
            
            # Record danger event in database
            if db_manager:
                db_manager.record_danger_event(
                    track_id=track_id,
                    event_type='no_helmet_in_danger_zone',
                    description=f'Person {track_id} entered danger zone without helmet'
                )
        
        # Update tracked object in database
        if db_manager:
            db_manager.update_tracked_object(track_id, has_helmet, in_danger_zone)
            
    else:
        # Update existing ID stats if state changed
        current_stats = id_stats[track_id]
        
        # Update helmet status
        if current_stats['has_helmet'] != has_helmet:
            current_stats['has_helmet'] = has_helmet
            
            # Record helmet removal event if in danger zone
            if not has_helmet and in_danger_zone and db_manager:
                db_manager.record_danger_event(
                    track_id=track_id,
                    event_type='helmet_removed_in_danger_zone',
                    description=f'Person {track_id} removed helmet while in danger zone'
                )
        
        # Update danger zone status
        if current_stats['in_danger_zone'] != in_danger_zone:
            current_stats['in_danger_zone'] = in_danger_zone
            
            # Count new danger event if entering danger zone without helmet
            if in_danger_zone and not has_helmet and not current_stats['danger_event_counted']:
                total_danger_events += 1
                current_stats['danger_event_counted'] = True
                
                # Record danger event in database
                if db_manager:
                    db_manager.record_danger_event(
                        track_id=track_id,
                        event_type='no_helmet_in_danger_zone',
                        description=f'Person {track_id} entered danger zone without helmet'
                    )
        
        # Update tracked object in database
        if db_manager:
            db_manager.update_tracked_object(track_id, has_helmet, in_danger_zone)

def get_current_statistics():
    """Calculate current statistics from tracked IDs."""
    global id_stats, total_danger_events
    
    persons_detected = len(processed_ids)
    people_with_helmets = sum(1 for stats in id_stats.values() if stats['has_helmet'])
    people_without_helmets = persons_detected - people_with_helmets
    people_in_danger_zone = sum(1 for stats in id_stats.values() if stats['in_danger_zone'])
    
    return {
        'persons_detected': persons_detected,
        'people_with_helmets': people_with_helmets,
        'people_without_helmets': people_without_helmets,
        'people_in_danger_zone': people_in_danger_zone,
        'total_danger_events': total_danger_events,
        'tracks_active': len(id_stats)
    }

def get_realtime_statistics():
    """Get real-time statistics from current frame."""
    global tracker
    
    if not tracker or not hasattr(tracker, 'id_has_helmet') or not hasattr(tracker, 'id_in_danger_zone'):
        return {
            'persons_detected': 0,
            'people_with_helmets': 0,
            'people_without_helmets': 0,
            'people_in_danger_zone': 0,
            'tracks_active': 0
        }
    
    # Get current active tracks (only tracks that are currently visible in frame)
    current_tracks = []
    for track_id in tracker.id_has_helmet.keys():
        # Only count tracks that are recently updated (within last few frames)
        # This ensures we only count people currently visible in the frame
        if track_id in tracker.id_has_helmet and track_id in tracker.id_in_danger_zone:
            current_tracks.append(track_id)
    
    persons_detected = len(current_tracks)
    
    people_with_helmets = 0
    people_without_helmets = 0
    people_in_danger_zone = 0
    
    for track_id in current_tracks:
        has_helmet = tracker.id_has_helmet.get(track_id, False)
        in_danger_zone = tracker.id_in_danger_zone.get(track_id, False)
        
        if has_helmet:
            people_with_helmets += 1
        else:
            people_without_helmets += 1
        
        if in_danger_zone:
            people_in_danger_zone += 1
    
    return {
        'persons_detected': persons_detected,
        'people_with_helmets': people_with_helmets,
        'people_without_helmets': people_without_helmets,
        'people_in_danger_zone': people_in_danger_zone,
        'tracks_active': persons_detected
    }

def camera_loop():
    """Main camera processing loop."""
    global output_frame, stats, is_running, roi_drawing_mode, roi_points, frames_processed, last_frame_time, db_manager
    
    while is_running:
        if camera is None or not camera.isOpened():
            time.sleep(0.03)
            continue
        
        ret, frame = camera.read()
        if not ret:
            continue
        frame = frame.copy()

        try:
            # Update frame processing time
            last_frame_time = time.time()
            
            # Draw ROI points if in drawing mode
            if roi_drawing_mode and roi_points:
                for i, point in enumerate(roi_points):
                    cv2.circle(frame, point, 5, (0, 255, 255), -1)
                    cv2.putText(frame, str(i+1), (point[0]+10, point[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw lines between points
                for i in range(len(roi_points)):
                    if i < len(roi_points) - 1:
                        cv2.line(frame, roi_points[i], roi_points[i+1], (0, 255, 255), 2)
                    else:
                        cv2.line(frame, roi_points[i], roi_points[0], (0, 255, 255), 2)
            
            # Process frame if tracker is available
            if tracker and not roi_drawing_mode:
                processed_frame, frame_stats = tracker.process_frame(
                    frame, 
                    draw_detections=True,
                    iou_threshold=DEFAULT_CONFIG['iou_threshold']
                )
                frames_processed += 1
                
                # Update statistics from tracker's internal state
                if tracker.id_has_helmet and tracker.id_in_danger_zone:
                    for track_id in tracker.id_has_helmet:
                        has_helmet = tracker.id_has_helmet.get(track_id, False)
                        in_danger_zone = tracker.id_in_danger_zone.get(track_id, False)
                        update_statistics_for_id(track_id, has_helmet, in_danger_zone)
                
                # Use frame_stats directly for current frame statistics
                current_frame_stats = {
                    'persons_detected': frame_stats.get('persons_detected', 0),
                    'people_with_helmets': frame_stats.get('people_with_helmets', 0),
                    'people_without_helmets': frame_stats.get('people_without_helmets', 0),
                    'people_in_danger_zone': frame_stats.get('people_in_danger_zone', 0),
                    'tracks_active': frame_stats.get('tracks_active', 0),
                    'total_danger_events': 0  # Reset to 0 for current frame only
                }
                
                # Update frame stats
                frame_stats.update(current_frame_stats)
                
                # Save frame statistics to database (every 30 frames to avoid too much data)
                if db_manager and frames_processed % 30 == 0:
                    db_manager.save_frame_statistics(current_frame_stats, frames_processed)
                
            else:
                processed_frame = frame
                frame_stats = get_realtime_statistics()
                frame_stats['total_danger_events'] = total_danger_events
                frames_processed += 1
            
            if processed_frame is not None:
                # Encode frame for web streaming
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if ret:
                    with lock:
                        output_frame = buffer.tobytes()
                        # Create a new stats dictionary to avoid type issues
                        stats = {
                            'frames_processed': frames_processed,
                            'last_frame_time': str(last_frame_time),  # Convert to string to avoid type issues
                            **frame_stats
                        }
                        
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Log error to database
            if db_manager:
                db_manager.log_system_event('error', f'Frame processing error: {e}')
            continue
        
        time.sleep(0.03)  # ~30 FPS

def generate_frames():
    """Generate video frames for web streaming."""
    while True:
        with lock:
            if output_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
        time.sleep(0.03)

def shutdown_server():
    """Gracefully shutdown the server."""
    global is_running, camera, db_manager, current_session_id
    print("\n🛑 서버를 종료합니다...")
    is_running = False
    
    if camera:
        camera.release()
    
    # End monitoring session
    if db_manager and current_session_id:
        db_manager.end_session(current_session_id)
        db_manager.log_system_event('info', 'Server shutdown initiated')
        db_manager.disconnect()
    
    # Force exit after 2 seconds
    def force_exit():
        time.sleep(2)
        os._exit(0)
    
    threading.Thread(target=force_exit).start()

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html', config=DEFAULT_CONFIG)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    """Get current statistics."""
    global stats, roi_drawing_mode, roi_points, is_running, frames_processed, last_frame_time
    
    # Get only real-time statistics from current frame
    if tracker and not roi_drawing_mode:
        # Use the stats from the last processed frame
        with lock:
            current_stats = {
                'persons_detected': stats.get('persons_detected', 0),
                'people_with_helmets': stats.get('people_with_helmets', 0),
                'people_without_helmets': stats.get('people_without_helmets', 0),
                'people_in_danger_zone': stats.get('people_in_danger_zone', 0),
                'tracks_active': stats.get('tracks_active', 0),
                'total_danger_events': 0,  # Reset to 0 for current frame only
                'is_running': is_running,
                'roi_drawing_mode': roi_drawing_mode,
                'roi_points_count': len(roi_points),
                'frames_processed': frames_processed,
                'last_frame_time': str(last_frame_time)
            }
    else:
        # No tracker or in drawing mode
        current_stats = {
            'persons_detected': 0,
            'people_with_helmets': 0,
            'people_without_helmets': 0,
            'people_in_danger_zone': 0,
            'tracks_active': 0,
            'total_danger_events': 0,
            'is_running': is_running,
            'roi_drawing_mode': roi_drawing_mode,
            'roi_points_count': len(roi_points),
            'frames_processed': frames_processed,
            'last_frame_time': str(last_frame_time)
        }
    
    # Determine system status
    if roi_drawing_mode:
        current_stats['system_status'] = 'drawing'
    elif is_running and (time.time() - last_frame_time) < 2.0:  # 2초 이내에 프레임이 처리되었으면 실행 중
        current_stats['system_status'] = 'running'
    else:
        current_stats['system_status'] = 'stopped'
    
    return jsonify(current_stats)

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """Get or update configuration."""
    global DEFAULT_CONFIG, tracker
    
    if request.method == 'POST':
        data = request.get_json()
        
        # Update configuration
        for key, value in data.items():
            if key in DEFAULT_CONFIG:
                DEFAULT_CONFIG[key] = value
        
        # Reinitialize tracker with new settings
        if tracker:
            try:
                tracker = UnifiedROITracker(
                    model_path=DEFAULT_CONFIG['model_path'],
                    conf_thresh=DEFAULT_CONFIG['conf_thresh'],
                    max_age=DEFAULT_CONFIG['max_age'],
                    device=DEFAULT_CONFIG['device'],
                    detection_interval=DEFAULT_CONFIG['detection_interval']
                )
                
                # Log configuration change
                if db_manager:
                    db_manager.log_system_event('info', 'Configuration updated', data)
                
                return jsonify({"status": "success", "message": "Configuration updated"})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)})
    
    return jsonify(DEFAULT_CONFIG)

@app.route('/api/control', methods=['POST'])
def control():
    """Control operations."""
    global is_running, camera_thread, tracker, roi_drawing_mode, roi_points, frames_processed, processed_ids, id_stats, total_danger_events, db_manager
    
    data = request.get_json()
    action = data.get('action')
    
    if action == 'start':
        if not is_running:
            is_running = True
            camera_thread = threading.Thread(target=camera_loop)
            camera_thread.start()
            
            # Log start event
            if db_manager:
                db_manager.log_system_event('info', 'Tracking started')
            
            return jsonify({"status": "success", "message": "Tracking started"})
        else:
            return jsonify({"status": "info", "message": "Already running"})
    
    elif action == 'stop':
        is_running = False
        if camera_thread:
            camera_thread.join()
        
        # Log stop event
        if db_manager:
            db_manager.log_system_event('info', 'Tracking stopped')
        
        return jsonify({"status": "success", "message": "Tracking stopped"})
    
    elif action == 'reset':
        if tracker:
            tracker.reset_tracker()
            frames_processed = 0
            # Reset statistics
            processed_ids.clear()
            id_stats.clear()
            total_danger_events = 0
            
            # Log reset event
            if db_manager:
                db_manager.log_system_event('info', 'Tracker reset')
            
            return jsonify({"status": "success", "message": "Tracker reset"})
    
    elif action == 'clear_roi':
        if tracker:
            tracker.clear_roi()
            roi_points = []
            roi_drawing_mode = False
            
            # Log ROI clear event
            if db_manager:
                db_manager.log_system_event('info', 'ROI cleared')
            
            return jsonify({"status": "success", "message": "ROI cleared"})
    
    elif action == 'shutdown':
        # Start shutdown in a separate thread
        threading.Thread(target=shutdown_server).start()
        return jsonify({"status": "success", "message": "Server shutting down..."})
    
    elif action == 'start_roi_drawing':
        roi_drawing_mode = True
        roi_points = []
        
        # Log ROI drawing start
        if db_manager:
            db_manager.log_system_event('info', 'ROI drawing mode started')
        
        return jsonify({"status": "success", "message": "ROI drawing mode started"})
    
    elif action == 'finish_roi_drawing':
        if len(roi_points) >= 3 and tracker:
            try:
                tracker.set_roi(roi_points)
                roi_drawing_mode = False
                
                # Save ROI configuration to database
                if db_manager:
                    db_manager.save_roi_configuration('User_Defined_ROI', roi_points)
                    db_manager.log_system_event('info', f'ROI set with {len(roi_points)} points')
                
                return jsonify({"status": "success", "message": f"ROI set with {len(roi_points)} points"})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)})
        else:
            roi_drawing_mode = False
            roi_points = []
            return jsonify({"status": "error", "message": "Need at least 3 points for ROI"})
    
    elif action == 'cancel_roi_drawing':
        roi_drawing_mode = False
        roi_points = []
        
        # Log ROI drawing cancellation
        if db_manager:
            db_manager.log_system_event('info', 'ROI drawing cancelled')
        
        return jsonify({"status": "success", "message": "ROI drawing cancelled"})
    
    return jsonify({"status": "error", "message": "Unknown action"})

@app.route('/api/roi', methods=['POST'])
def set_roi():
    """Set ROI programmatically."""
    global tracker, db_manager
    
    data = request.get_json()
    points = data.get('points', [])
    
    if tracker and len(points) >= 3:
        try:
            tracker.set_roi(points)
            
            # Save ROI configuration to database
            if db_manager:
                db_manager.save_roi_configuration('Programmatic_ROI', points)
                db_manager.log_system_event('info', f'ROI set programmatically with {len(points)} points')
            
            return jsonify({"status": "success", "message": f"ROI set with {len(points)} points"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    
    return jsonify({"status": "error", "message": "Invalid ROI points"})

@app.route('/api/roi_click', methods=['POST'])
def roi_click():
    """Handle ROI drawing clicks."""
    global roi_points, roi_drawing_mode
    
    if not roi_drawing_mode:
        return jsonify({"status": "error", "message": "ROI drawing mode not active"})
    
    data = request.get_json()
    x = data.get('x')
    y = data.get('y')
    
    if x is not None and y is not None:
        roi_points.append((int(x), int(y)))
        return jsonify({
            "status": "success", 
            "message": f"Point {len(roi_points)} added at ({x}, {y})",
            "points_count": len(roi_points)
        })
    
    return jsonify({"status": "error", "message": "Invalid coordinates"})

@app.route('/api/database/stats')
def get_database_stats():
    """Get database statistics."""
    global db_manager, current_session_id
    
    if not db_manager:
        return jsonify({"status": "error", "message": "Database not connected"})
    
    try:
        # Get session summary
        session_summary = None
        if current_session_id:
            session_summary = db_manager.get_session_summary(current_session_id)
        
        # Get recent danger events
        recent_events = db_manager.get_recent_danger_events(hours=24)
        
        return jsonify({
            "status": "success",
            "session_summary": session_summary,
            "recent_events": recent_events[:10],  # Last 10 events
            "session_id": current_session_id
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Initialize database
    db_initialized = initialize_database()
    
    # Initialize camera and tracker
    if initialize_camera():
        print("Camera and tracker initialized successfully")
        
        # Create monitoring session if database is available
        if db_initialized:
            create_monitoring_session()
        
        # Start camera loop
        is_running = True
        camera_thread = threading.Thread(target=camera_loop)
        camera_thread.start()
        
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("Failed to initialize camera") 