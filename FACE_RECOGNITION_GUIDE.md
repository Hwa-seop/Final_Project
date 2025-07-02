# 얼굴 인식 기반 ID 부여 기능 사용 가이드

## 📋 개요

이 시스템은 얼굴 인식을 통해 개인을 식별하고 고유한 ID를 부여하는 기능을 제공합니다. 기존의 헬멧 감지 및 ROI 추적 기능과 함께 작동하여 더욱 정확한 개인별 안전 모니터링이 가능합니다.

## 🚀 주요 기능

### 1. 얼굴 인식 및 ID 부여
- **자동 얼굴 감지**: DeepFace 라이브러리를 사용한 정확한 얼굴 감지
- **고유 ID 생성**: 감지된 얼굴에 대해 `Person_1`, `Person_2` 등의 고유 ID 부여
- **얼굴 매칭**: 기존에 등록된 얼굴과의 유사도 기반 매칭
- **지속적 추적**: 프레임 간 얼굴 인식 결과 유지

### 2. 통합 안전 모니터링
- **헬멧 착용 상태**: 얼굴 ID와 연결된 헬멧 착용 여부 추적
- **위험구역 진입**: 얼굴 ID별 위험구역 진입 이벤트 기록
- **개인별 통계**: 얼굴 ID별 안전 준수율 및 위험 이벤트 통계

### 3. 데이터 저장 및 관리
- **얼굴 데이터 저장**: 얼굴 임베딩을 `approved_face_deepface.pkl` 파일에 저장
- **데이터베이스 연동**: 얼굴 인식 이벤트를 MySQL 데이터베이스에 저장
- **자동 정리**: 오래된 얼굴 데이터 자동 정리

## 🛠️ 설치 및 설정

### 1. 필요한 라이브러리 설치

```bash
# DeepFace 라이브러리 설치
pip install deepface==0.0.79

# 기타 필요한 라이브러리 설치
pip install -r requirements_flask.txt
```

### 2. 실행 스크립트 사용

```bash
# 얼굴 인식 기능이 포함된 앱 실행
./run_face_recognition_app.sh
```

### 3. 수동 실행

```bash
# 얼굴 인식 테스트
python3 test_face_recognition.py

# Flask 웹 애플리케이션 실행
python3 flask_app_optimized.py
```

## 📊 시스템 구성

### 얼굴 인식 설정

```python
# 얼굴 인식 관련 설정
FACE_RECOGNITION_INTERVAL = 30      # 얼굴 인식 간격 (프레임)
FACE_SIMILARITY_THRESHOLD = 0.6     # 얼굴 유사도 임계값
MAX_FACE_EMBEDDINGS = 50            # 최대 얼굴 임베딩 저장 수
```

### 메모리 관리 설정

```python
# 메모리 관리 설정
MAX_OBJECTS = 10                    # 최대 추적 객체 수
MAX_MEMORY_PERCENT = 80             # 최대 메모리 사용률 (%)
MEMORY_CHECK_INTERVAL = 100         # 메모리 체크 간격 (프레임)
OBJECT_CLEANUP_INTERVAL = 50        # 객체 정리 간격 (프레임)
```

## 🎯 사용 방법

### 1. 얼굴 인식 테스트

```bash
python3 test_face_recognition.py
```

**테스트 기능:**
- 실시간 얼굴 감지 및 인식
- 새로운 얼굴 자동 등록
- 기존 얼굴과의 매칭
- 얼굴 데이터 저장 (`s` 키)

### 2. 웹 인터페이스 사용

1. **애플리케이션 시작**
   ```bash
   python3 flask_app_optimized.py
   ```

2. **웹 브라우저 접속**
   - URL: `http://localhost:5000`
   - 실시간 비디오 스트림 확인
   - 얼굴 인식 결과 시각화

3. **API 엔드포인트**
   - `/api/stats`: 전체 통계 정보
   - `/api/faces`: 등록된 얼굴 정보
   - `/api/face_recognition/toggle`: 얼굴 인식 기능 토글

### 3. 얼굴 인식 결과 확인

**시각적 표시:**
- 파란색 박스: 감지된 얼굴 영역
- 얼굴 ID 라벨: `Face: Person_1 (0.85)` 형태
- 유사도 점수: 0.0~1.0 범위

**통계 정보:**
- 등록된 얼굴 수
- 얼굴 인식 활성화 상태
- 개인별 안전 준수율

## 🔧 고급 설정

### 1. 얼굴 인식 정확도 조정

```python
# 유사도 임계값 조정 (높을수록 더 엄격한 매칭)
FACE_SIMILARITY_THRESHOLD = 0.7  # 기본값: 0.6

# 얼굴 인식 간격 조정 (낮을수록 더 자주 실행)
FACE_RECOGNITION_INTERVAL = 15   # 기본값: 30
```

### 2. 메모리 사용량 최적화

```python
# 최대 얼굴 임베딩 수 조정
MAX_FACE_EMBEDDINGS = 30         # 기본값: 50

# 객체 정리 간격 조정
OBJECT_CLEANUP_INTERVAL = 30     # 기본값: 50
```

### 3. 얼굴 인식 모델 변경

```python
# DeepFace 모델 변경 (extract_face_embedding 함수에서)
embedding = DeepFace.represent(face_img, model_name="VGG-Face", enforce_detection=False)
# 사용 가능한 모델: "Facenet", "VGG-Face", "OpenFace", "DeepID", "ArcFace"
```

## 📈 성능 최적화

### 1. 하드웨어 권장사항

**저사양 시스템 (CPU만 사용):**
- 얼굴 인식 간격: 60프레임
- 최대 객체 수: 10개
- 프레임 크기: 320x240

**중간 사양 시스템:**
- 얼굴 인식 간격: 30프레임
- 최대 객체 수: 20개
- 프레임 크기: 640x480

**고사양 시스템 (GPU 사용):**
- 얼굴 인식 간격: 15프레임
- 최대 객체 수: 50개
- 프레임 크기: 1280x720

### 2. 메모리 사용량 모니터링

```bash
# 실시간 메모리 사용량 확인
python3 memory_monitor.py
```

### 3. 성능 튜닝 팁

1. **얼굴 인식 간격 조정**: 시스템 성능에 따라 15~60프레임 범위에서 조정
2. **유사도 임계값 조정**: 환경에 따라 0.5~0.8 범위에서 조정
3. **프레임 크기 조정**: 성능과 정확도의 균형점 찾기
4. **객체 수 제한**: 메모리 사용량과 처리 속도 고려

## 🐛 문제 해결

### 1. 얼굴 인식이 작동하지 않는 경우

**확인사항:**
- DeepFace 라이브러리 설치 여부
- 카메라 접근 권한
- 충분한 조명 조건

**해결방법:**
```bash
# DeepFace 재설치
pip uninstall deepface
pip install deepface==0.0.79

# 카메라 권한 확인
ls -l /dev/video*
```

### 2. 메모리 사용량이 높은 경우

**확인사항:**
- 동시 추적 객체 수
- 얼굴 임베딩 저장 수
- 프레임 크기

**해결방법:**
```python
# 설정값 조정
MAX_OBJECTS = 10
MAX_FACE_EMBEDDINGS = 20
MAX_FRAME_SIZE = (320, 240)
```

### 3. 얼굴 인식 정확도가 낮은 경우

**확인사항:**
- 조명 조건
- 얼굴 각도
- 유사도 임계값

**해결방법:**
```python
# 유사도 임계값 낮추기
FACE_SIMILARITY_THRESHOLD = 0.5

# 얼굴 인식 간격 줄이기
FACE_RECOGNITION_INTERVAL = 15
```

## 📝 API 참조

### 얼굴 인식 관련 API

**얼굴 정보 조회**
```http
GET /api/faces
```

**응답 예시:**
```json
{
  "total_faces": 3,
  "faces": [
    {
      "id": "Person_1",
      "first_seen": 1640995200.0,
      "last_seen": 1640995260.0,
      "detection_count": 15
    }
  ]
}
```

**얼굴 인식 토글**
```http
POST /api/face_recognition/toggle
Content-Type: application/json

{
  "enabled": true
}
```

## 🔒 보안 고려사항

1. **개인정보 보호**: 얼굴 데이터는 로컬에만 저장
2. **데이터 암호화**: 민감한 정보는 암호화하여 저장
3. **접근 제어**: 관리자 권한으로만 시스템 설정 변경
4. **로그 관리**: 얼굴 인식 이벤트 로그 보관 및 관리

## 📞 지원

문제가 발생하거나 추가 기능이 필요한 경우:
1. 로그 파일 확인
2. 시스템 요구사항 점검
3. 설정값 조정
4. 개발팀에 문의

---

**버전**: 1.0  
**최종 업데이트**: 2024년 12월  
**작성자**: AI Safety Monitoring Team 