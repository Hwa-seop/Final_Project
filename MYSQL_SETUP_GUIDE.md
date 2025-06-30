# MySQL 데이터베이스 설정 가이드

## 개요
AI 안전 관제 시스템이 MySQL 데이터베이스를 사용하도록 변경되었습니다. 이 가이드는 MySQL 설정 및 사용법을 설명합니다.

## 🗄️ 데이터베이스 구조

### 핵심 테이블 (6개)
1. **monitoring_sessions** - 모니터링 세션 정보
2. **frame_statistics** - 프레임별 통계 데이터
3. **tracked_objects** - 추적된 객체 정보
4. **danger_events** - 위험 상황 이벤트
5. **roi_configurations** - ROI(위험 구역) 설정
6. **system_logs** - 시스템 로그

### 제거된 테이블
- `users` - 사용자 관리 (불필요)
- 복잡한 권한 시스템 제거
- 트리거 및 함수 제거

## 🚀 설치 및 설정

### 1. MySQL 설치
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install mysql-server

# CentOS/RHEL
sudo yum install mysql-server
```

### 2. MySQL 서비스 시작
```bash
sudo systemctl start mysql
sudo systemctl enable mysql
```

### 3. MySQL 보안 설정
```bash
sudo mysql_secure_installation
```

### 4. 데이터베이스 초기화
```bash
# 스크립트 실행 권한 부여
chmod +x init_mysql_db.sh

# 데이터베이스 및 테이블 생성
./init_mysql_db.sh
```

### 5. 환경 변수 설정
```bash
# env_example.txt를 .env로 복사하고 수정
cp env_example.txt .env

# .env 파일 편집
nano .env
```

`.env` 파일 예시:
```env
# MySQL Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_NAME=ai_safety_monitor
DB_USER=root
DB_PASSWORD=your_password_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=1
```

## 📦 의존성 설치

### Python 패키지 설치
```bash
pip install -r requirements_flask.txt
```

### 주요 변경사항
- `psycopg2-binary` → `mysql-connector-python` 변경
- `deep-sort-realtime` 추가
- `flask-cors` 추가

## 🔧 데이터베이스 연결 테스트

### Python으로 연결 테스트
```python
from database_manager import DatabaseManager

# 데이터베이스 매니저 생성
db_manager = DatabaseManager()

# 연결 테스트
if db_manager.connect():
    print("✅ MySQL 연결 성공!")
    db_manager.disconnect()
else:
    print("❌ MySQL 연결 실패!")
```

### MySQL 클라이언트로 연결 테스트
```bash
mysql -u root -p ai_safety_monitor
```

## 📊 데이터베이스 관리

### 테이블 확인
```sql
USE ai_safety_monitor;
SHOW TABLES;
```

### 세션 통계 조회
```sql
SELECT * FROM session_summary;
```

### 최근 위험 이벤트 조회
```sql
SELECT * FROM danger_events 
ORDER BY event_time DESC 
LIMIT 10;
```

### 데이터 정리
```sql
-- 오래된 세션 삭제 (30일 이상)
DELETE FROM monitoring_sessions 
WHERE start_time < DATE_SUB(NOW(), INTERVAL 30 DAY);

-- 오래된 로그 삭제 (7일 이상)
DELETE FROM system_logs 
WHERE timestamp < DATE_SUB(NOW(), INTERVAL 7 DAY);
```

## 🚀 애플리케이션 실행

### Flask 앱 실행
```bash
python flask_app.py
```

### 웹 브라우저 접속
```
http://localhost:5000
```

## 🔍 문제 해결

### 일반적인 문제들

#### 1. 연결 거부 오류
```bash
# MySQL 서비스 상태 확인
sudo systemctl status mysql

# MySQL 재시작
sudo systemctl restart mysql
```

#### 2. 인증 오류
```bash
# MySQL에 root로 접속
sudo mysql

# 새 사용자 생성 및 권한 부여
CREATE USER 'ai_user'@'localhost' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON ai_safety_monitor.* TO 'ai_user'@'localhost';
FLUSH PRIVILEGES;
```

#### 3. 데이터베이스 없음 오류
```bash
# 데이터베이스 생성
mysql -u root -p -e "CREATE DATABASE ai_safety_monitor;"
```

#### 4. 테이블 없음 오류
```bash
# 테이블 생성 스크립트 실행
mysql -u root -p ai_safety_monitor < create_database.sql
```

## 📈 성능 최적화

### 인덱스 확인
```sql
SHOW INDEX FROM frame_statistics;
SHOW INDEX FROM tracked_objects;
SHOW INDEX FROM danger_events;
```

### 쿼리 성능 분석
```sql
EXPLAIN SELECT * FROM frame_statistics WHERE session_id = 1;
```

### 데이터베이스 상태 확인
```sql
SHOW STATUS LIKE 'Connections';
SHOW STATUS LIKE 'Threads_connected';
```

## 🔄 백업 및 복원

### 데이터베이스 백업
```bash
mysqldump -u root -p ai_safety_monitor > backup_$(date +%Y%m%d).sql
```

### 데이터베이스 복원
```bash
mysql -u root -p ai_safety_monitor < backup_20231201.sql
```

## 📝 로그 확인

### MySQL 로그
```bash
sudo tail -f /var/log/mysql/error.log
```

### 애플리케이션 로그
```bash
# Flask 앱 실행 시 로그 확인
python flask_app.py 2>&1 | tee app.log
```

## 🎯 주요 개선사항

1. **단순화된 구조**: 불필요한 테이블 제거
2. **MySQL 최적화**: MySQL 특화 기능 활용
3. **자동 업데이트**: `ON UPDATE CURRENT_TIMESTAMP` 사용
4. **캐스케이드 삭제**: 세션 삭제 시 관련 데이터 자동 삭제
5. **성능 향상**: 적절한 인덱스 설정

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. MySQL 서비스 상태
2. 데이터베이스 연결 설정
3. 테이블 존재 여부
4. 사용자 권한
5. 애플리케이션 로그 