#!/bin/bash

# MySQL 데이터베이스 초기화 스크립트
# AI 안전 관제 시스템용 데이터베이스 및 테이블 생성

echo "🚀 MySQL 데이터베이스 초기화 시작..."

# MySQL 접속 정보 (환경 변수에서 가져오거나 기본값 사용)
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-3306}
DB_USER=${DB_USER:-root}
DB_PASSWORD=${DB_PASSWORD:-}
DB_NAME=${DB_NAME:-ai_safety_monitor}

echo "📊 데이터베이스 정보:"
echo "   Host: $DB_HOST"
echo "   Port: $DB_PORT"
echo "   User: $DB_USER"
echo "   Database: $DB_NAME"

# 데이터베이스 생성
echo "📝 데이터베이스 생성 중..."
if [ -z "$DB_PASSWORD" ]; then
    mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -e "CREATE DATABASE IF NOT EXISTS $DB_NAME CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
else
    mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" -e "CREATE DATABASE IF NOT EXISTS $DB_NAME CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
fi

if [ $? -eq 0 ]; then
    echo "✅ 데이터베이스 생성 완료"
else
    echo "❌ 데이터베이스 생성 실패"
    exit 1
fi

# 테이블 생성
echo "📋 테이블 생성 중..."
if [ -z "$DB_PASSWORD" ]; then
    mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" "$DB_NAME" < create_database.sql
else
    mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" "$DB_NAME" < create_database.sql
fi

if [ $? -eq 0 ]; then
    echo "✅ 테이블 생성 완료"
else
    echo "❌ 테이블 생성 실패"
    exit 1
fi

echo "🎉 MySQL 데이터베이스 초기화 완료!"
echo ""
echo "📋 생성된 테이블:"
echo "   - monitoring_sessions (모니터링 세션)"
echo "   - frame_statistics (프레임별 통계)"
echo "   - tracked_objects (추적 객체)"
echo "   - danger_events (위험 이벤트)"
echo "   - roi_configurations (ROI 설정)"
echo "   - system_logs (시스템 로그)"
echo ""
echo "🔧 다음 단계:"
echo "   1. 환경 변수 설정 (env_example.txt 참조)"
echo "   2. Flask 앱 실행: python flask_app.py"
echo "   3. 웹 브라우저에서 http://localhost:5000 접속" 