#!/usr/bin/env python3
"""
API 라우트 모듈

Flask 웹 애플리케이션의 API 엔드포인트를 정의합니다.
"""

from flask import Flask, request, jsonify, Response, render_template
import json
import time
from typing import Dict, Any, Optional
from camera_manager import CameraManager
from database_manager import DatabaseManager
from config import Config

def create_app(camera_manager: CameraManager, db_manager: DatabaseManager) -> Flask:
    """
    Flask 애플리케이션을 생성하고 라우트를 등록합니다.
    
    Args:
        camera_manager: 카메라 관리자 인스턴스
        db_manager: 데이터베이스 관리자 인스턴스
        
    Returns:
        Flask: 설정된 Flask 애플리케이션
    """
    app = Flask(__name__)
    
    # 전역 변수로 매니저들 저장
    app.camera_manager = camera_manager
    app.db_manager = db_manager
    
    @app.route('/')
    def index():
        """메인 페이지를 렌더링합니다."""
        return render_template('index.html')
    
    @app.route('/video_feed')
    def video_feed():
        """비디오 스트림을 제공합니다."""
        def generate():
            while True:
                frame = app.camera_manager.get_output_frame()
                if frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.03)  # 30 FPS
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/api/status')
    def get_status():
        """시스템 상태를 반환합니다."""
        try:
            camera_status = app.camera_manager.get_status()
            detector_stats = app.camera_manager.get_detector_stats()
            
            return jsonify({
                'success': True,
                'camera_status': camera_status,
                'detector_stats': detector_stats,
                'timestamp': time.time()
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/roi/start', methods=['POST'])
    def start_roi_drawing():
        """ROI 그리기 모드를 시작합니다."""
        try:
            app.camera_manager.start_roi_drawing()
            app.db_manager.log_system_event('INFO', 'ROI 그리기 모드 시작', 'api_routes')
            
            return jsonify({
                'success': True,
                'message': 'ROI 그리기 모드가 시작되었습니다.'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/roi/add_point', methods=['POST'])
    def add_roi_point():
        """ROI 점을 추가합니다."""
        try:
            data = request.get_json()
            x = data.get('x')
            y = data.get('y')
            
            if x is None or y is None:
                return jsonify({
                    'success': False,
                    'error': 'x, y 좌표가 필요합니다.'
                }), 400
            
            success = app.camera_manager.add_roi_point(int(x), int(y))
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'점 추가됨: ({x}, {y})',
                    'points_count': len(app.camera_manager.roi_points)
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'ROI 그리기 모드가 활성화되지 않았습니다.'
                }), 400
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/roi/finish', methods=['POST'])
    def finish_roi_drawing():
        """ROI 그리기를 완료합니다."""
        try:
            data = request.get_json()
            roi_name = data.get('name', f'ROI_{int(time.time())}')
            
            success = app.camera_manager.finish_roi_drawing()
            
            if success:
                # 데이터베이스에 ROI 설정 저장
                app.db_manager.save_roi_configuration(roi_name, app.camera_manager.roi_points)
                app.db_manager.log_system_event('INFO', f'ROI 설정 저장됨: {roi_name}', 'api_routes')
                
                return jsonify({
                    'success': True,
                    'message': f'ROI 설정이 완료되었습니다: {roi_name}',
                    'roi_name': roi_name,
                    'points_count': len(app.camera_manager.roi_points)
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'ROI 설정을 완료할 수 없습니다. 최소 3개의 점이 필요합니다.'
                }), 400
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/roi/cancel', methods=['POST'])
    def cancel_roi_drawing():
        """ROI 그리기를 취소합니다."""
        try:
            app.camera_manager.cancel_roi_drawing()
            app.db_manager.log_system_event('INFO', 'ROI 그리기 취소됨', 'api_routes')
            
            return jsonify({
                'success': True,
                'message': 'ROI 그리기가 취소되었습니다.'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/roi/clear', methods=['POST'])
    def clear_roi():
        """현재 ROI를 초기화합니다."""
        try:
            app.camera_manager.clear_roi()
            app.db_manager.log_system_event('INFO', 'ROI 초기화됨', 'api_routes')
            
            return jsonify({
                'success': True,
                'message': 'ROI가 초기화되었습니다.'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/roi/load', methods=['GET'])
    def load_roi():
        """저장된 ROI 설정을 로드합니다."""
        try:
            roi_config = app.db_manager.get_active_roi_configuration()
            
            if roi_config:
                app.camera_manager.set_roi(roi_config['points'])
                return jsonify({
                    'success': True,
                    'roi_config': roi_config
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '저장된 ROI 설정이 없습니다.'
                }), 404
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/detector/reset', methods=['POST'])
    def reset_detector():
        """헬멧 감지기를 리셋합니다."""
        try:
            app.camera_manager.reset_detector()
            app.db_manager.log_system_event('INFO', '헬멧 감지기 리셋됨', 'api_routes')
            
            return jsonify({
                'success': True,
                'message': '헬멧 감지기가 리셋되었습니다.'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/config/update', methods=['POST'])
    def update_config():
        """설정을 업데이트합니다."""
        try:
            data = request.get_json()
            
            # 설정 업데이트
            Config.update_config(data)
            app.camera_manager.update_config(data)
            
            app.db_manager.log_system_event('INFO', f'설정 업데이트됨: {data}', 'api_routes')
            
            return jsonify({
                'success': True,
                'message': '설정이 업데이트되었습니다.',
                'config': Config.get_config()
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/config/get')
    def get_config():
        """현재 설정을 반환합니다."""
        try:
            return jsonify({
                'success': True,
                'config': Config.get_config()
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/events/recent')
    def get_recent_events():
        """최근 위험 이벤트들을 반환합니다."""
        try:
            limit = request.args.get('limit', 50, type=int)
            events = app.db_manager.get_recent_danger_events(limit)
            
            return jsonify({
                'success': True,
                'events': events,
                'count': len(events)
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/events/resolve', methods=['POST'])
    def resolve_event():
        """위험 이벤트를 해결됨으로 표시합니다."""
        try:
            data = request.get_json()
            event_id = data.get('event_id')
            
            if event_id is None:
                return jsonify({
                    'success': False,
                    'error': 'event_id가 필요합니다.'
                }), 400
            
            success = app.db_manager.mark_event_resolved(event_id)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'이벤트 {event_id}가 해결됨으로 표시되었습니다.'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '이벤트를 업데이트할 수 없습니다.'
                }), 500
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/statistics/summary')
    def get_statistics_summary():
        """통계 요약을 반환합니다."""
        try:
            hours = request.args.get('hours', 24, type=int)
            summary = app.db_manager.get_statistics_summary(hours)
            
            return jsonify({
                'success': True,
                'summary': summary
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/camera/start', methods=['POST'])
    def start_camera():
        """카메라를 시작합니다."""
        try:
            if not app.camera_manager.is_running:
                success = app.camera_manager.initialize_camera()
                if success:
                    app.camera_manager.start_camera_loop()
                    app.db_manager.log_system_event('INFO', '카메라 시작됨', 'api_routes')
                    
                    return jsonify({
                        'success': True,
                        'message': '카메라가 시작되었습니다.'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': '카메라 초기화에 실패했습니다.'
                    }), 500
            else:
                return jsonify({
                    'success': False,
                    'error': '카메라가 이미 실행 중입니다.'
                }), 400
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/camera/stop', methods=['POST'])
    def stop_camera():
        """카메라를 중지합니다."""
        try:
            if app.camera_manager.is_running:
                app.camera_manager.stop_camera_loop()
                app.db_manager.log_system_event('INFO', '카메라 중지됨', 'api_routes')
                
                return jsonify({
                    'success': True,
                    'message': '카메라가 중지되었습니다.'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '카메라가 실행 중이 아닙니다.'
                }), 400
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.errorhandler(404)
    def not_found(error):
        """404 에러 핸들러"""
        return jsonify({
            'success': False,
            'error': '요청한 리소스를 찾을 수 없습니다.'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """500 에러 핸들러"""
        return jsonify({
            'success': False,
            'error': '내부 서버 오류가 발생했습니다.'
        }), 500
    
    return app 