# 🤖 AI 안전 모니터링 시스템

실시간 헬멧 감지 및 ROI(위험 구역) 추적 시스템을 웹 브라우저에서 제어하고 모니터링할 수 있는 Flask 기반 웹 애플리케이션입니다.

## 📋 프로젝트 목적

공사장, 작업실 등 헬멧 착용 / 위험 구역 진입 제한이 필요한 곳에서 사용하기 위함입니다.

## ✨ 주요 기능

### 1. 실시간 헬멧 감지
- **YOLOv5 기반**: 미리 학습된 PT 파일을 사용하여 헬멧 착용/미착용 여부 감지
- **DeepSORT 추적**: 사람별 고유 ID 할당 및 추적
- **IoU 기반 판단**: 머리 영역과 헬멧 영역의 겹침 정도로 정확한 판단

### 2. 위험 구역(ROI) 설정 및 모니터링
- **웹 UI에서 직접 설정**: 마우스 클릭으로 ROI 영역 그리기
- **실시간 위험 감지**: 헬멧 미착용자가 위험 구역 진입 시 자동 감지
- **시각적 표시**: 위험 구역을 빨간색 폴리곤으로 표시

### 3. 웹 기반 제어 및 모니터링
- **실시간 비디오 스트림**: MJPEG 형식으로 웹 브라우저에 전송
- **통계 대시보드**: 실시간 감지 통계 및 시스템 상태 표시
- **이벤트 기록**: 위험 이벤트 자동 기록 및 조회

### 4. 데이터베이스 연동
- **MySQL 연동**: 위험 이벤트, ROI 설정, 시스템 로그 저장
- **통계 누적**: 프레임별 통계 기록 및 누적 위험 이벤트 카운팅
- **이벤트 관리**: 위험 이벤트 해결 표시 및 조회

## 🛠️ 사용 기술

- **YOLOv5**: 객체(사람/헬멧/미착용자) 감지
- **DeepSORT**: 객체 ID 기반 추적
- **OpenCV**: 프레임 처리 및 시각화
- **Flask**: 웹 서버 및 RESTful API
- **MySQL**: 데이터 저장 및 세션 관리

## 📁 프로젝트 구조

```
FF/
├── flask_app.py              # 메인 애플리케이션
├── config.py                 # 설정 관리
├── helmet_detector.py        # 헬멧 감지 및 추적
├── camera_manager.py         # 카메라 관리
├── database_manager.py       # 데이터베이스 관리
├── api_routes.py            # Flask API 라우트
├── requirements.txt         # Python 라이브러리 목록
├── install.bat              # Windows 설치 스크립트
├── install.sh               # Linux/Mac 설치 스크립트
├── templates/
│   └── index.html           # 웹 인터페이스
├── best.pt                  # YOLO 모델 파일 (사용자 제공)
└── README.md               # 프로젝트 설명서
```

## 🚀 설치 및 실행

### 1. 사전 요구사항

- **Python 3.8 이상**
- **MySQL 5.7 이상** (선택사항, 없으면 데이터베이스 기능 제한)
- **웹캠 또는 IP 카메라**
- **YOLO 모델 파일** (`best.pt`)

### 2. 자동 설치

#### Windows
```bash
# 설치 스크립트 실행
install.bat
```

#### Linux/Mac
```bash
# 실행 권한 부여
chmod +x install.sh

# 설치 스크립트 실행
./install.sh
```

### 3. 수동 설치

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 라이브러리 설치
pip install -r requirements.txt
```

### 4. 데이터베이스 설정 (선택사항)

```sql
-- MySQL에서 데이터베이스 생성
CREATE DATABASE ai_safety_monitor CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 사용자 생성 및 권한 부여
CREATE USER 'safety_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON ai_safety_monitor.* TO 'safety_user'@'localhost';
FLUSH PRIVILEGES;
```

### 5. 환경 변수 설정 (선택사항)

```bash
# Windows
set DB_HOST=localhost
set DB_PORT=3306
set DB_NAME=ai_safety_monitor
set DB_USER=safety_user
set DB_PASSWORD=your_password

# Linux/Mac
export DB_HOST=localhost
export DB_PORT=3306
export DB_NAME=ai_safety_monitor
export DB_USER=safety_user
export DB_PASSWORD=your_password
```

### 6. 애플리케이션 실행

```bash
# 가상환경 활성화
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 애플리케이션 실행
python flask_app.py
```

### 7. 웹 접속

브라우저에서 `http://localhost:5000`으로 접속

## 🎯 사용 방법

### 1. 카메라 시작
- 웹 인터페이스에서 "카메라 시작" 버튼 클릭
- 실시간 비디오 스트림 확인

### 2. ROI 설정
1. "ROI 그리기 시작" 버튼 클릭
2. 비디오 화면에서 점들을 클릭하여 위험 구역 설정
3. 최소 3개 이상의 점을 찍은 후 "ROI 설정 완료" 클릭
4. ROI 이름 입력 후 저장

### 3. 모니터링
- 실시간으로 헬멧 착용 여부 감지
- 위험 구역 진입 시 자동 이벤트 기록
- 통계 대시보드에서 실시간 상태 확인

### 4. 이벤트 관리
- "최근 위험 이벤트" 섹션에서 이벤트 조회
- 이벤트 해결 표시 및 관리

## 🔧 설정 옵션

### 기본 설정 (config.py에서 수정 가능)

```python
DEFAULT_CONFIG = {
    'model_path': 'best.pt',        # YOLO 모델 파일 경로
    'conf_thresh': 0.2,             # 신뢰도 임계값
    'iou_threshold': 0.1,           # IoU 임계값
    'max_age': 30,                  # 트랙 최대 유지 시간
    'detection_interval': 5,        # 감지 간격 (프레임 단위)
    'device': 'auto',               # 실행 디바이스 (auto/cuda/cpu)
    'source': 0                     # 카메라 소스 (0=기본 카메라)
}
```

### 웹 인터페이스에서 설정 변경
- 설정 조회 버튼으로 현재 설정 확인
- API를 통해 실시간 설정 변경 가능

## 📊 API 엔드포인트

### 시스템 제어
- `GET /api/status` - 시스템 상태 조회
- `POST /api/camera/start` - 카메라 시작
- `POST /api/camera/stop` - 카메라 중지
- `POST /api/detector/reset` - 감지기 리셋

### ROI 관리
- `POST /api/roi/start` - ROI 그리기 시작
- `POST /api/roi/add_point` - ROI 점 추가
- `POST /api/roi/finish` - ROI 설정 완료
- `POST /api/roi/cancel` - ROI 그리기 취소
- `POST /api/roi/clear` - ROI 초기화
- `GET /api/roi/load` - 저장된 ROI 로드

### 데이터 조회
- `GET /api/events/recent` - 최근 위험 이벤트 조회
- `GET /api/statistics/summary` - 통계 요약 조회
- `POST /api/events/resolve` - 이벤트 해결 표시

### 설정 관리
- `GET /api/config/get` - 설정 조회
- `POST /api/config/update` - 설정 업데이트

## 🐛 문제 해결

### 일반적인 문제

1. **카메라가 열리지 않는 경우**
   - 카메라가 다른 프로그램에서 사용 중인지 확인
   - `config.py`에서 `source` 값을 변경 (1, 2, ...)

2. **모델 로드 오류**
   - `best.pt` 파일이 올바른 경로에 있는지 확인
   - GPU 메모리 부족 시 `device: 'cpu'`로 설정

3. **데이터베이스 연결 오류**
   - MySQL 서비스가 실행 중인지 확인
   - 데이터베이스 접속 정보 확인
   - 데이터베이스 없이도 기본 기능 사용 가능

4. **성능 문제**
   - `detection_interval` 값을 늘려서 성능 향상
   - GPU 사용 시 CUDA 설치 확인

### 로그 확인

```bash
# 애플리케이션 실행 시 콘솔에서 로그 확인
python flask_app.py

# 데이터베이스 로그 확인 (MySQL)
SELECT * FROM system_logs ORDER BY timestamp DESC LIMIT 10;
```

## 🔄 업데이트 및 유지보수

### 정기적인 업데이트
- 라이브러리 업데이트: `pip install --upgrade -r requirements.txt`
- 모델 파일 업데이트: 새로운 `best.pt` 파일로 교체

### 백업 및 복구
- 데이터베이스 백업: `mysqldump ai_safety_monitor > backup.sql`
- 설정 백업: `config.py` 파일 복사

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.

---

**주의**: 이 시스템은 안전 모니터링을 위한 보조 도구입니다. 실제 안전 관리는 전문 인력과 적절한 안전 절차에 따라 수행해야 합니다. 