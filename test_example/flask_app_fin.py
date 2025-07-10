#!/usr/bin/env python3
"""
이 Flask 앱은 실시간 헬멧 검출 및 ROI 추적을 위한 웹 인터페이스를 제공합니다.
사용자는 웹 브라우저를 통해 실시간 비디오 스트림을 보고, 설정을 제어하고, 통계를 모니터링할 수 있습니다.

주요 기능:
- 실시간 비디오 스트리밍
- YOLOv5 기반 헬멧 검출
- DeepSORT 기반 객체 추적
- ROI 기반 위험 구역 모니터링
- 웹 인터페이스를 통한 제어
- MySQL 데이터베이스 연동
"""
# 기본 모듈 및 외부 의존성
from flask import Flask, render_template, Response, jsonify, request
import cv2
from functools import lru_cache
import os
import time
import threading
import subprocess
import signal
import sys
import json
from tcp_helmet_module import HelmetController

helmet_controller = HelmetController()
helmet_controller.start_server()

from unified_roi_tracker_module_fin import UnifiedROITracker
from database_manager_patched import DatabaseManager, init_database, get_database_manager
from face_unified import FaceUnified  # FaceUnified 모듈 경로에 맞게 조정
face_unified = FaceUnified() 

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

track_id_to_name = {} # 트랙 ID와 이름 매핑 (예: 헬멧 착용 여부 등)


# ID 기반 추적 및 통계 관리
# Statistics tracking for unique IDs
processed_ids = set()   # 처리된 ID 모음
id_stats = {}  # ID별 통계 정보를 저장하기 위한 딕셔너리
total_danger_events = 0  # 총 위험 이벤트 카운터 (누적)

# Database manager
db_manager = None
current_session_id = None

# 카메라 모드 변수
FFMPEG_MODE = False  # FFMPEG 모드 (FFmpeg를 통한 스트리밍)
MJPEG_MODE = True  # MJPEG 모드 (Flask를 통한 MJPEG 스트리밍)

# Wan 송출 모드
# LOCALTUNNEL_MODE = False  # 로컬 터널 모드 (로컬에서 실행 시)
# NGROK = False  # Ngrok 모드 (외부 접근을 위한 Ngrok 사용)

if FFMPEG_MODE:
    ffmpeg = subprocess.Popen([
        'ffmpeg',
        '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', '640x480', '-r', '30',
        '-i', '-',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
        '-f', 'flv', 'rtmp://localhost/live/stream'
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def monitor_ffmpeg_errors():
        while True:
            output = ffmpeg.stderr.readline()
            if output == '' and ffmpeg.poll() is not None:
                break
            if output:
                print(f"[FFmpeg STDERR] {output.strip()}")

    threading.Thread(target=monitor_ffmpeg_errors, daemon=True).start()

# === [기본 설정값 정의] ===
# GStreamer configuration
GSTREAMER_CONFIG = {
    # ───────── V4L2 카메라 (일반적 환경) ─────────
    'webcam': (
        'v4l2src device=/dev/video0 ! videoconvert ! video/x-raw,width=640,height=480,framerate=30/1 ! appsink'
    ),
    'usb_camera': (
        'v4l2src device=/dev/video1 ! videoconvert ! '
        'video/x-raw,width=1280,height=720,framerate=30/1 ! appsink'
    ),
    # ───────── 나머지 소스는 기존과 동일 ─────────
    'rtsp': (
        'rtspsrc location=rtsp://192.168.1.100:554/stream ! '
        'rtph264depay ! h264parse ! avdec_h264 ! '
        'videoconvert ! appsink'
    ),
    'file': (
        'filesrc location=/path/to/video.mp4 ! '
        'decodebin ! videoconvert ! appsink'
    ),
    'test': (
        'videotestsrc ! '
        'video/x-raw,width=640,height=480,framerate=30/1 ! '
        'videoconvert ! appsink'
    ),
}

DEFAULT_CONFIG = {
    'model_path': '/home/hwaseop/final/best.pt',
    'conf_thresh': 0.3,
    'iou_threshold': 0.2,
    'max_age': 60,
    'detection_interval': 3,
    'device': 'auto',
    'source': 'webcam',  # 반드시 'webcam'으로 설정
}

# Configuration cache (단순화)
_config_cache = None

def get_cached_config():
    global _config_cache
    if _config_cache is None:
        _config_cache = DEFAULT_CONFIG.copy()
    return _config_cache

# === [DB 연결 초기화] ===
def initialize_database():
    """
    데이터베이스 연결을 초기화합니다.
    
    환경 변수에서 데이터베이스 설정을 읽어와 연결을 시도합니다.
    연결 실패 시 데이터베이스 없이 실행됩니다.
    
    Returns:
        bool: 데이터베이스 연결 성공 여부
    """
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
            print("✅ 데이터베이스 연결 성공")
            return True
        else:
            print("❌ 데이터베이스 연결 실패 - 데이터베이스 없이 실행")
            return False
    except Exception as e:
        print(f"❌ 데이터베이스 초기화 오류: {e} - 데이터베이스 없이 실행")
        return False
# === [세션 생성 (DB 기록용)] ===
def create_monitoring_session():
    """
    새로운 모니터링 세션을 생성합니다.
    
    데이터베이스에 새로운 세션을 생성하고 세션 ID를 반환합니다.
    세션은 위험 이벤트 기록 및 통계 추적에 사용됩니다.
    
    Returns:
        int or None: 생성된 세션 ID 또는 None (실패 시)
    """
    global db_manager, current_session_id
    
    if not db_manager:
        return None
    
    try:
        session_name = f"AI_Safety_Monitor_{int(time.time())}"
        current_session_id = db_manager.create_session(session_name, DEFAULT_CONFIG)
        
        if current_session_id:
            print(f"✅ 모니터링 세션 생성: {session_name} (ID: {current_session_id})")
            db_manager.log_system_event('info', 'Monitoring session started', {
                'session_id': current_session_id,
                'config': DEFAULT_CONFIG
            })
        
        return current_session_id
    except Exception as e:
        print(f"❌ 모니터링 세션 생성 실패: {e}")
        return None

# === [카메라 및 트래커 초기화] ===
def initialize_camera():
    """
    카메라와 트래커를 초기화합니다.
    GStreamer 파이프라인만 시도합니다. 실패 시 에러 출력 후 종료.
    Returns:
        bool: 초기화 성공 여부
    """
    global camera, tracker
    try:
        config = get_cached_config()
        source = config.get('source', 'webcam')
        # GStreamer 파이프라인만 시도
        if source in GSTREAMER_CONFIG:
            gstreamer_pipeline = GSTREAMER_CONFIG[source]
            print(f"[DEBUG] 사용 파이프라인: {gstreamer_pipeline}")
            camera = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
            if camera and camera.isOpened():
                ret, test_frame = camera.read()
                if ret and test_frame is not None:
                    print(f"✅ GStreamer 파이프라인 '{source}'에서 카메라를 성공적으로 열었습니다")
                else:
                    print(f"❌ GStreamer 파이프라인 '{source}'에서 프레임을 읽을 수 없습니다")
                    camera.release()
                    camera = None
            else:
                print(f"❌ GStreamer 파이프라인 '{source}'을 열 수 없습니다")
                camera = None
        else:
            print(f"❌ '{source}'는(은) 등록된 GStreamer 파이프라인이 아닙니다. GSTREAMER_CONFIG를 확인하세요.")
            camera = None
        if not camera or not camera.isOpened():
            print("❌ GStreamer 파이프라인으로 카메라를 열 수 없습니다. 프로그램을 종료합니다.")
            return False
        # Initialize tracker
        tracker = UnifiedROITracker(
            model_path=DEFAULT_CONFIG['model_path'],
            conf_thresh=DEFAULT_CONFIG['conf_thresh'],
            max_age=DEFAULT_CONFIG['max_age'],
            device=DEFAULT_CONFIG['device'],
            detection_interval=DEFAULT_CONFIG['detection_interval'],
            threshold=0.5            
        )
        return True
    except Exception as e:
        print(f"❌ 카메라 초기화 오류: {e}")
        return False

def update_statistics_for_id(track_id, has_helmet, in_danger_zone):
    """
    특정 트랙 ID의 통계를 업데이트합니다. 중복 카운팅을 방지합니다.
    
    Args:
        track_id (int): 추적 객체의 고유 ID
        has_helmet (bool): 헬멧 착용 여부
        in_danger_zone (bool): 위험 구역 내 위치 여부
    """
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
    """
    추적된 ID들로부터 현재 통계를 계산합니다.
    
    Returns:
        dict: 현재 통계 정보
    """
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
    """
    현재 프레임에서의 실시간 통계를 가져옵니다.
    
    Returns:
        dict: 실시간 통계 정보
    """
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
    """
    메인 카메라 처리 루프입니다.
    
    카메라에서 프레임을 읽어와서 객체 검출, 추적, ROI 분석을 수행합니다.
    처리된 프레임을 웹 스트리밍용으로 인코딩하고 통계를 업데이트합니다.
    """
    global output_frame, stats, is_running, roi_drawing_mode, roi_points, frames_processed, last_frame_time, db_manager
    global MJPEG_MODE, FFMPEG_MODE
    
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
                            if in_danger_zone and not has_helmet:
                                helmet_controller.set_helmet_status("removed")  # 경고
                            elif has_helmet:
                                helmet_controller.set_helmet_status("wearing")  # 정상 상태

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
            
            if MJPEG_MODE:
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
            if FFMPEG_MODE:
                if ffmpeg:
                    try:
                        ffmpeg.stdin.write(processed_frame.tobytes())
                    except Exception as e:
                        print(f"[FFmpeg 송출 오류] {e}")
                        FFMPEG_MODE = False
            if MJPEG_MODE and FFMPEG_MODE:
                raise ValueError("MJPEG_MODE와 FFMPEG_MODE는 동시에 True일 수 없습니다.")            
                        
        except Exception as e:
            print(f"❌ 프레임 처리 오류: {e}")
            # Log error to database
            if db_manager:
                db_manager.log_system_event('error', f'Frame processing error: {e}')
            continue
        
        time.sleep(0.03)  # ~30 FPS

def generate_frames():
    """
    웹 스트리밍을 위한 비디오 프레임을 생성합니다.
    
    Yields:
        bytes: JPEG 인코딩된 프레임 데이터
    """
    while True:
        with lock:
            if output_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
        time.sleep(0.03)

def shutdown_server():
    """
    서버를 안전하게 종료합니다.
    
    카메라를 해제하고, 데이터베이스 연결을 종료하며, 모니터링 세션을 끝냅니다.
    """
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
    """
    메인 페이지를 렌더링합니다.
    
    Returns:
        str: HTML 템플릿 렌더링 결과
    """
    return render_template('index.html', config=DEFAULT_CONFIG)

@app.route('/video_feed')
def video_feed():
    """
    비디오 스트리밍 라우트입니다.
    MJPEG 또는 FFMPEG 스트리밍 모드에 따라 프레임을 생성합니다.
    Returns:
        Response: MJPEG 스트림 응답
        Response: FFMPEG 스트림 응답
    """
    if MJPEG_MODE:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif FFMPEG_MODE:
        # FFMPEG 스트림을 위한 라우트
        return Response(ffmpeg.stdout, mimetype='video/mp4')
    else:
        return Response("Streaming mode not enabled.", status=400)

@app.route('/api/stats')
def get_stats():
    """
    현재 통계 정보를 반환합니다.
    
    Returns:
        JSON: 현재 통계 정보
    """
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
    """
    설정을 가져오거나 업데이트합니다.
    
    GET: 현재 설정 반환
    POST: 새로운 설정으로 업데이트
    
    Returns:
        JSON: 설정 정보 또는 업데이트 결과
    """
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
    """
    시스템 제어 작업을 수행합니다.
    
    지원하는 액션:
    - start: 트래킹 시작
    - stop: 트래킹 정지
    - reset: 트래커 리셋
    - clear_roi: ROI 지우기
    - shutdown: 서버 종료
    - start_roi_drawing: ROI 그리기 모드 시작
    - finish_roi_drawing: ROI 그리기 완료
    - cancel_roi_drawing: ROI 그리기 취소
    
    Returns:
        JSON: 작업 결과
    """
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
    """
    프로그래밍 방식으로 ROI를 설정합니다.
    
    Returns:
        JSON: ROI 설정 결과
    """
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
    """
    ROI 그리기 클릭을 처리합니다.
    
    Returns:
        JSON: 클릭 처리 결과
    """
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
    """
    데이터베이스 통계를 반환합니다.
    
    Returns:
        JSON: 데이터베이스 통계 정보
    """
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

@app.route('/api/register_face', methods=['POST'])
def register_face():
    """
    사용자 얼굴을 등록합니다.
    JSON 형식: { "username": "worker1" }
    """
    data = request.get_json()
    username = data.get("username")

    if not username:
        return jsonify({"status": "error", "message": "Username is required."})

    try:
        success = face_unified.register_face_with_camera(username)
        if success:
            return jsonify({"status": "success", "message": f"{username} 등록 완료"})
        else:
            return jsonify({"status": "error", "message": f"{username} 등록 실패"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/upload_face', methods=['GET', 'POST'])
def upload_face():
    if request.method == 'POST':
        username = request.form.get('username')
        file = request.files.get('image')

        if not username or not file:
            return "이름과 이미지를 모두 입력하세요.", 400

        save_dir = os.path.join(app.root_path, "faces")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{username}.jpg")
        file.save(save_path)

        try:
            face_unified.face_manager.register_face(username, save_path)
            return f"{username} 등록 완료"
        except Exception as e:
            return f"등록 실패: {e}", 500

    return render_template("upload_face.html")
    
if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Initialize database
    db_initialized = initialize_database()
    
    # Initialize camera and tracker
    if initialize_camera():
        print("✅ 카메라와 트래커가 성공적으로 초기화되었습니다")
        
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
        print("❌ 카메라 초기화 실패") 


        
