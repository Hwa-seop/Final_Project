#!/bin/bash

# Unified ROI Tracker Flask Web Application Launcher
# 이 스크립트는 Flask 웹 애플리케이션을 실행합니다.

echo "🛡️ Unified ROI Tracker Flask Web Application"
echo "=============================================="

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Python 환경 확인
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3가 설치되어 있지 않습니다."
    exit 1
fi

echo "✅ Python3 확인됨: $(python3 --version)"

# 가상환경 확인 및 생성
if [ ! -d "venv" ]; then
    echo "📦 가상환경을 생성합니다..."
    python3 -m venv venv
fi

# 가상환경 활성화
echo "🔧 가상환경을 활성화합니다..."
source venv/bin/activate

# 의존성 설치
echo "📚 필요한 패키지를 설치합니다..."
pip install --upgrade pip
pip install -r requirements_flask.txt

# templates 디렉토리 확인
if [ ! -d "templates" ]; then
    echo "📁 templates 디렉토리를 생성합니다..."
    mkdir -p templates
fi

# 모델 파일 확인
MODEL_PATH="/home/lws/kulws2025/kubig2025/final_project/yolov5/helmet_detection/helmet_detection/weights/best.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  모델 파일을 찾을 수 없습니다: $MODEL_PATH"
    echo "   올바른 경로로 flask_app.py를 수정해주세요."
fi

# 카메라 권한 확인
if [ -e "/dev/video0" ]; then
    echo "📹 카메라 장치 확인됨: /dev/video0"
else
    echo "⚠️  카메라 장치를 찾을 수 없습니다."
    echo "   웹캠이 연결되어 있는지 확인해주세요."
fi

# 종료 핸들러 함수
cleanup() {
    echo ""
    echo "🛑 종료 신호를 받았습니다. 정리 중..."
    
    # Flask 프로세스 종료
    if [ ! -z "$FLASK_PID" ]; then
        echo "📱 Flask 웹 애플리케이션을 종료합니다..."
        kill -TERM "$FLASK_PID" 2>/dev/null
        
        # 5초 대기 후 강제 종료
        sleep 5
        kill -KILL "$FLASK_PID" 2>/dev/null
    fi
    
    # 카메라 리소스 정리
    echo "📹 카메라 리소스를 정리합니다..."
    
    # 가상환경 비활성화
    deactivate 2>/dev/null
    
    echo "👋 Unified ROI Tracker가 안전하게 종료되었습니다."
    exit 0
}

# 시그널 핸들러 등록
trap cleanup SIGINT SIGTERM

echo ""
echo "🚀 Flask 웹 애플리케이션을 시작합니다..."
echo "   웹 브라우저에서 http://localhost:5000 으로 접속하세요."
echo "   종료하려면 Ctrl+C를 누르거나 웹 인터페이스의 종료 버튼을 사용하세요."
echo ""

# Flask 앱 실행 (백그라운드에서)
python3 flask_app.py &
FLASK_PID=$!

# Flask 프로세스가 시작될 때까지 대기
sleep 3

# 프로세스가 실행 중인지 확인
if kill -0 "$FLASK_PID" 2>/dev/null; then
    echo "✅ Flask 웹 애플리케이션이 성공적으로 시작되었습니다."
    echo "   프로세스 ID: $FLASK_PID"
    echo ""
    echo "📋 사용 가능한 기능:"
    echo "   • 실시간 비디오 스트리밍"
    echo "   • 헬멧 검출 및 추적"
    echo "   • ROI 직접 그리기 (클릭으로 점 추가)"
    echo "   • 실시간 통계 모니터링 (현재 프레임 기준, 위험 이벤트는 누적)"
    echo "   • 웹 기반 설정 조정 (검출 간격: 5/7/10/15프레임)"
    echo "   • 안전한 서버 종료"
    echo ""
    echo "🔄 프로세스를 모니터링합니다. 종료하려면 Ctrl+C를 누르세요..."
    echo "   웹 인터페이스에서 '정지' 버튼을 누르면 시스템이 정지됩니다."
    echo "   '시작' 버튼을 누르면 다시 실행됩니다."
    echo "   '리셋' 버튼을 누르면 통계가 초기화됩니다."
    echo "   검출 간격을 조정하여 성능과 정확도를 균형있게 설정할 수 있습니다."
    echo "   실시간 통계는 현재 프레임의 상태를 반영하며, 총 위험 이벤트는 누적됩니다."
    echo ""
    
    # 프로세스 상태 모니터링
    while kill -0 "$FLASK_PID" 2>/dev/null; do
        sleep 2
        # 프로세스가 여전히 살아있는지 확인
        if ! kill -0 "$FLASK_PID" 2>/dev/null; then
            echo "⚠️  Flask 프로세스가 예기치 않게 종료되었습니다."
            break
        fi
    done
    
    echo "🛑 Flask 프로세스가 종료되었습니다."
else
    echo "❌ Flask 웹 애플리케이션 시작에 실패했습니다."
    exit 1
fi

# 정리 함수 호출 (정상 종료 시에도)
cleanup 