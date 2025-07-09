import socket
import threading
import json
import time

class HelmetController:
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        self.esp_conn = None
        self.server = None
        self.is_running = False
        self.server_thread = None
        self.helmet_status = "unknown"  # "wearing", "removed", "unknown"
        self.status_callback = None
        
    def start_server(self):
        """서버를 시작합니다."""
        if self.is_running:
            print("서버가 이미 실행 중입니다.")
            return
            
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        self.is_running = True
        
        print(f"헬멧 제어 서버 실행 중... (TCP {self.host}:{self.port})")
        print("ESP8266 헬멧 연결을 기다리는 중...")
        
        # 서버 스레드 시작
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
    
    def _server_loop(self):
        """서버 메인 루프"""
        while self.is_running:
            try:
                print("연결 대기중...")
                if self.server:  # None 체크 추가
                    conn, addr = self.server.accept()
                    print(f"[ESP8266] 헬멧 연결됨: {addr}")
                    threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
            except Exception as e:
                if self.is_running:
                    print(f"서버 오류: {e}")
    
    def _handle_client(self, conn, addr):
        """클라이언트 연결 처리"""
        self.esp_conn = conn
        print(f"[ESP8266] 헬멧 연결됨: {addr}")
        print("헬멧이 연결되었습니다. 딥러닝 모델로 헬멧 착용 여부를 판단하여 제어할 수 있습니다.")
        
        try:
            while self.is_running:
                # 헬멧에서 보내는 데이터가 있으면 수신 (선택사항)
                try:
                    conn.settimeout(1.0)  # 1초 타임아웃
                    data = conn.recv(1024)
                    if data:
                        received_data = data.decode().strip()
                        print(f"[ESP8266] 수신: {received_data}")
                except socket.timeout:
                    # 타임아웃은 정상적인 상황
                    pass
                except Exception as e:
                    if self.is_running:
                        print(f"데이터 수신 오류: {e}")
                    break
                    
        except Exception as e:
            print(f"[ESP8266] 연결 해제: {addr}, 예외: {e}")
        finally:
            conn.close()
            self.esp_conn = None
            print("헬멧 연결이 해제되었습니다.")
    
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
        return self.esp_conn is not None
    
    def get_helmet_status(self):
        """현재 헬멧 상태 반환"""
        return self.helmet_status
    
    def send_command(self, command):
        """헬멧에 명령 전송"""
        if not self.is_connected():
            print("ESP8266 헬멧이 연결되어 있지 않습니다.")
            return False
        
        try:
            # JSON 형태로 명령 전송
            if self.esp_conn:
                json_command = json.dumps(command) + '\n'
                self.esp_conn.sendall(json_command.encode('utf-8'))
                print(f"헬멧에 명령 전송: {command}")
                return True
            else:
                print("ESP8266 헬멧 연결이 없습니다.")
                return False
        except Exception as e:
            print(f"명령 전송 실패: {e}")
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
    
    def stop_server(self):
        """서버 중지"""
        self.is_running = False
        if self.esp_conn:
            self.esp_conn.close()
        if self.server:
            self.server.close()
        print("헬멧 제어 서버가 중지되었습니다.")

# 딥러닝 모델과 연동하는 예시
def main():
    # 헬멧 상태 변경 콜백 함수
    def on_helmet_status_change(status, message):
        if status == "removed":
            print("🚨 딥러닝 모델이 헬멧 벗음을 감지했습니다!")
            print("   - LED와 부저가 켜져서 사용자에게 알림을 보냅니다.")
        elif status == "wearing":
            print("✅ 딥러닝 모델이 헬멧 착용을 확인했습니다!")
            print("   - LED와 부저가 꺼져서 알림을 해제합니다.")
    
    # 헬멧 컨트롤러 생성
    helmet = HelmetController()
    
    # 상태 변경 콜백 설정
    helmet.set_status_callback(on_helmet_status_change)
    
    try:
        # 서버 시작
        helmet.start_server()
        
        print("\n=== 딥러닝 모델 연동 가이드 ===")
        print("1. ESP8266 헬멧이 연결되면 '헬멧이 연결되었습니다' 메시지가 출력됩니다.")
        print("2. 딥러닝 모델에서 헬멧 착용 여부를 판단한 후:")
        print("   helmet.set_helmet_status('removed')  # 헬멧 벗음 감지")
        print("   helmet.set_helmet_status('wearing')  # 헬멧 착용 감지")
        print("3. 자동으로 헬멧에 LED/부저 제어 명령이 전송됩니다.")
        
        # 서버가 계속 실행되도록 대기
        while helmet.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n서버를 종료합니다...")
        helmet.stop_server()

if __name__ == "__main__":
    main()