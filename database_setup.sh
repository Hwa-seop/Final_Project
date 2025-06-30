#!/bin/bash

# AI 안전 관제 시스템 데이터베이스 설정 스크립트
# 이 스크립트는 PostgreSQL 데이터베이스를 설정합니다.

echo "🗄️ AI 안전 관제 시스템 데이터베이스 설정"
echo "=============================================="

# PostgreSQL 설치 확인
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL이 설치되어 있지 않습니다."
    echo "   Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
    echo "   CentOS/RHEL: sudo yum install postgresql postgresql-server"
    echo "   macOS: brew install postgresql"
    exit 1
fi

echo "✅ PostgreSQL 확인됨: $(psql --version)"

# 데이터베이스 설정
DB_NAME="ai_safety_monitor"
DB_USER="postgres"
DB_PASSWORD=""

# 사용자 입력 받기
echo ""
echo "📝 데이터베이스 설정을 입력하세요:"
read -p "데이터베이스 이름 (기본값: $DB_NAME): " input_db_name
DB_NAME=${input_db_name:-$DB_NAME}

read -p "데이터베이스 사용자 (기본값: $DB_USER): " input_db_user
DB_USER=${input_db_user:-$DB_USER}

read -s -p "데이터베이스 비밀번호: " DB_PASSWORD
echo ""

# PostgreSQL 서비스 시작
echo ""
echo "🔄 PostgreSQL 서비스를 시작합니다..."
sudo systemctl start postgresql
sudo systemctl enable postgresql

# 데이터베이스 생성
echo ""
echo "🗄️ 데이터베이스를 생성합니다..."
sudo -u postgres psql -c "CREATE DATABASE $DB_NAME;" 2>/dev/null || echo "데이터베이스가 이미 존재합니다."

# 사용자 권한 설정
echo ""
echo "👤 사용자 권한을 설정합니다..."
sudo -u postgres psql -c "ALTER USER $DB_USER WITH PASSWORD '$DB_PASSWORD';" 2>/dev/null || echo "사용자 권한 설정 완료"

# 테이블 생성
echo ""
echo "📋 테이블을 생성합니다..."
sudo -u postgres psql -d $DB_NAME -f create_database.sql

# 연결 테스트
echo ""
echo "🔗 데이터베이스 연결을 테스트합니다..."
PGPASSWORD=$DB_PASSWORD psql -h localhost -U $DB_USER -d $DB_NAME -c "SELECT version();" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ 데이터베이스 연결 성공!"
    
    # 환경 변수 파일 생성
    echo ""
    echo "📄 환경 변수 파일을 생성합니다..."
    cat > .env << EOF
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=$DB_NAME
DB_USER=$DB_USER
DB_PASSWORD=$DB_PASSWORD

# Application Configuration
FLASK_ENV=production
FLASK_DEBUG=False
EOF
    
    echo "✅ .env 파일이 생성되었습니다."
    echo ""
    echo "📋 설정 완료!"
    echo "   데이터베이스: $DB_NAME"
    echo "   사용자: $DB_USER"
    echo "   호스트: localhost"
    echo ""
    echo "🚀 이제 run_flask.sh를 실행하여 애플리케이션을 시작할 수 있습니다."
    
else
    echo "❌ 데이터베이스 연결 실패!"
    echo "   설정을 확인하고 다시 시도해주세요."
    exit 1
fi 