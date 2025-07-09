import requests
import json
import time
import threading
from helmet_config import ESP8266_CONFIG, DEBUG

class HTTPHelmetController:
    def __init__(self, esp_ip=None, esp_port=None):
        # 설정 파일에서 기본값 사용
        self.esp_ip = esp_ip or ESP8266_CONFIG['ip_address']
        self.esp_port = esp_port or ESP8266_CONFIG['port']
        self.base_url = f"http://{self.esp_ip}:{self.esp_port}"
        self.helmet_status = "unknown"  # "wearing", "removed", "unknown"
        self.status_callback = None
        self._connected = False
        self.connection_check_thread = None
        self.is_running = False
        
    def start_connection_monitor(self):
        """연결 상태를 주기적으로 확인하는 스레드 시작"""
        self.is_running = True
        self.connection_check_thread = threading.Thread(target=self._connection_monitor_loop, daemon=True)
        self.connection_check_thread.start()
        print(f"HTTP 헬멧 컨트롤러 시작: {self.base_url}")
    
    def _connection_monitor_loop(self):
        """연결 상태 모니터링 루프"""
        while self.is_running:
            try:
                # ESP8266 상태 확인
                response = requests.get(f"{self.base_url}/status", timeout=ESP8266_CONFIG['timeout'])
                if response.status_code == 200:
                    if not self._connected:
                        self._connected = True
                        if DEBUG['show_connection_status']:
                            print("✅ ESP8266 헬멧 연결됨")
                else:
                    if self._connected:
                        self._connected = False
                        if DEBUG['show_connection_status']:
                            print("❌ ESP8266 헬멧 연결 끊어짐")
            except requests.exceptions.RequestException:
                if self._connected:
                    self._connected = False
                    if DEBUG['show_connection_status']:
                        print("❌ ESP8266 헬멧 연결 끊어짐")
            
            time.sleep(ESP8266_CONFIG['check_interval'])  # 설정된 간격으로 연결 상태 확인
    
    def set_helmet_status(self, status):
        """
        딥러닝 모델의 추론 결과로 헬멧 상태 설정
        헬멧 착용 여부에 따라 자동으로 LED/부저 제어
        
        Args:
            status: "wearing" 또는 "removed"
        """
        if status not in ["wearing", "removed"]:
            print(f"잘못된 헬멧 상태: {status}")
            return False
        
        # 상태가 변경된 경우에만 처리
        if self.helmet_status == status:
            return True
        
        old_status = self.helmet_status
        self.helmet_status = status
        
        print(f"헬멧 상태 변경: {old_status} -> {status}")
        
        if status == "removed":
            # 헬멧 벗음 감지 - 긴급 알림
            success = self.emergency_alert()
            if success and self.status_callback:
                self.status_callback("removed", "긴급 알림 활성화")
            return success
            
        elif status == "wearing":
            # 헬멧 착용 감지 - 알림 해제
            success = self.clear_alert()
            if success and self.status_callback:
                self.status_callback("wearing", "알림 해제")
            return success
    
    def set_status_callback(self, callback_func):
        """헬멧 상태 변경 시 호출될 콜백 함수 설정"""
        self.status_callback = callback_func
    
    def is_connected(self):
        """ESP8266 헬멧 연결 상태 확인"""
        return self._connected
    
    def get_helmet_status(self):
        """현재 헬멧 상태 반환"""
        return self.helmet_status
    
    def send_command(self, command):
        """헬멧에 HTTP 명령 전송"""
        if not self._connected:
            print("ESP8266 헬멧이 연결되어 있지 않습니다.")
            return False
        
        try:
            # HTTP POST 요청으로 명령 전송
            url = f"{self.base_url}/control"
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, json=command, headers=headers, timeout=ESP8266_CONFIG['timeout'])
            
            if response.status_code == 200:
                if DEBUG['show_http_requests']:
                    print(f"헬멧에 명령 전송 성공: {command}")
                return True
            else:
                print(f"헬멧 명령 전송 실패: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"헬멧 명령 전송 실패: {e}")
            return False
    
    def emergency_alert(self):
        """긴급 알림 (LED와 부저 동시 작동)"""
        return self.send_command({"led": 1, "buzzer": 1})
    
    def clear_alert(self):
        """알림 해제 (LED와 부저 동시 끄기)"""
        return self.send_command({"led": 0, "buzzer": 0})
    
    def led_on(self):
        """LED 켜기"""
        return self.send_command({"led": 1})
    
    def led_off(self):
        """LED 끄기"""
        return self.send_command({"led": 0})
    
    def buzzer_on(self):
        """부저 켜기"""
        return self.send_command({"buzzer": 1})
    
    def buzzer_off(self):
        """부저 끄기"""
        return self.send_command({"buzzer": 0})
    
    def custom_command(self, led=None, buzzer=None):
        """사용자 정의 명령"""
        command = {}
        if led is not None:
            command["led"] = led
        if buzzer is not None:
            command["buzzer"] = buzzer
        return self.send_command(command)
    
    def get_esp_status(self):
        """ESP8266 상태 정보 가져오기"""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=ESP8266_CONFIG['timeout'])
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except requests.exceptions.RequestException:
            return None
    
    def stop_connection_monitor(self):
        """연결 모니터링 중지"""
        self.is_running = False
        if self.connection_check_thread:
            self.connection_check_thread.join(timeout=1)
        print("HTTP 헬멧 컨트롤러가 중지되었습니다.")

# 테스트용 메인 함수
def main():
    # 헬멧 상태 변경 콜백 함수
    def on_helmet_status_change(status, message):
        if status == "removed":
            print("🚨 딥러닝 모델이 헬멧 벗음을 감지했습니다!")
            print("   - LED와 부저가 켜져서 사용자에게 알림을 보냅니다.")
        elif status == "wearing":
            print("✅ 딥러닝 모델이 헬멧 착용을 확인했습니다!")
            print("   - LED와 부저가 꺼져서 알림을 해제합니다.")
    
    # HTTP 헬멧 컨트롤러 생성 (ESP8266 IP 주소 설정)
    helmet = HTTPHelmetController(esp_ip="192.168.1.100", esp_port=80)
    
    # 상태 변경 콜백 설정
    helmet.set_status_callback(on_helmet_status_change)
    
    try:
        # 연결 모니터링 시작
        helmet.start_connection_monitor()
        
        print("\n=== HTTP 헬멧 제어 가이드 ===")
        print("1. ESP8266 헬멧이 연결되면 'ESP8266 헬멧 연결됨' 메시지가 출력됩니다.")
        print("2. 딥러닝 모델에서 헬멧 착용 여부를 판단한 후:")
        print("   helmet.set_helmet_status('removed')  # 헬멧 벗음 감지")
        print("   helmet.set_helmet_status('wearing')  # 헬멧 착용 감지")
        print("3. HTTP 요청으로 헬멧에 LED/부저 제어 명령이 전송됩니다.")
        
        # 서버가 계속 실행되도록 대기
        while helmet.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n서버를 종료합니다...")
        helmet.stop_connection_monitor()

if __name__ == "__main__":
    main() 