# 헬멧 제어 설정 파일

# ESP8266 설정
ESP8266_CONFIG = {
    'ip_address': '192.168.0.113',  # ESP8266 IP 주소 (실제 IP로 변경 필요)
    'port': 80,                     # HTTP 포트
    'timeout': 3,                   # HTTP 요청 타임아웃 (초)
    'check_interval': 5,            # 연결 상태 확인 간격 (초)
}

# 헬멧 제어 명령
HELMET_COMMANDS = {
    'emergency_alert': {
        'led': 1,
        'buzzer': 1,
        'description': '긴급 알림 - LED와 부저 동시 작동'
    },
    'clear_alert': {
        'led': 0,
        'buzzer': 0,
        'description': '알림 해제 - LED와 부저 동시 끄기'
    },
    'led_on': {
        'led': 1,
        'description': 'LED 켜기'
    },
    'led_off': {
        'led': 0,
        'description': 'LED 끄기'
    },
    'buzzer_on': {
        'buzzer': 1,
        'description': '부저 켜기'
    },
    'buzzer_off': {
        'buzzer': 0,
        'description': '부저 끄기'
    }
}

# 헬멧 상태 매핑
HELMET_STATUS_MAPPING = {
    'wearing': {
        'description': '헬멧 착용',
        'action': 'clear_alert',
        'led': 0,
        'buzzer': 0
    },
    'removed': {
        'description': '헬멧 벗음',
        'action': 'emergency_alert',
        'led': 1,
        'buzzer': 1
    },
    'unknown': {
        'description': '상태 불명',
        'action': None,
        'led': 0,
        'buzzer': 0
    }
}

# 디버그 설정
DEBUG = {
    'enable_logging': True,
    'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'show_http_requests': True,
    'show_connection_status': True
}

def get_esp_url():
    """ESP8266 URL 반환"""
    return f"http://{ESP8266_CONFIG['ip_address']}:{ESP8266_CONFIG['port']}"

def get_command(command_name):
    """명령 이름으로 명령 객체 반환"""
    return HELMET_COMMANDS.get(command_name, {})

def get_status_mapping(status):
    """헬멧 상태로 매핑 정보 반환"""
    return HELMET_STATUS_MAPPING.get(status, HELMET_STATUS_MAPPING['unknown'])

def update_esp_ip(new_ip):
    """ESP8266 IP 주소 업데이트"""
    ESP8266_CONFIG['ip_address'] = new_ip
    print(f"ESP8266 IP 주소가 {new_ip}로 업데이트되었습니다.")

def print_config():
    """현재 설정 출력"""
    print("=== 헬멧 제어 설정 ===")
    print(f"ESP8266 IP: {ESP8266_CONFIG['ip_address']}")
    print(f"ESP8266 Port: {ESP8266_CONFIG['port']}")
    print(f"Timeout: {ESP8266_CONFIG['timeout']}초")
    print(f"Check Interval: {ESP8266_CONFIG['check_interval']}초")
    print(f"Debug Logging: {DEBUG['enable_logging']}")
    print(f"Log Level: {DEBUG['log_level']}")
    print("=====================") 