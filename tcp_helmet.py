import socket
import threading

HOST = "0.0.0.0"
PORT = 8000

esp_conn = None

def handle_client(conn, addr):
    global esp_conn
    esp_conn = conn
    print(f"[ESP8266] 연결됨: {addr}")
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            print(f"[ESP826] 상태 수신: {data.decode().strip()}")
    except Exception as e:
        print(f"[ESP8266] 연결 해제: {addr}, 예외: {e}")
    finally:
        conn.close()
        esp_conn = None

def cli():
    global esp_conn
    menu = (
        "\n--- 제어 메뉴 ---\n"
        "1: LED ON\n"
        "2: LED OFF\n"
        "3: BUZZER ON\n"
        "4: BUZZER OFF\n"
        "5: 종료\n"
        "----------------"
    )
    while True:
        print(menu)
        cmd = input("명령 입력: ").strip()
        if esp_conn:
            if cmd == "1":
                esp_conn.sendall(b'{"led":1}\n')
                print("LED ON 신호 전송")
            elif cmd == "2":
                esp_conn.sendall(b'{"led":0}\n')
                print("LED OFF 신호 전송")
            elif cmd == "3":
                esp_conn.sendall(b'{"buzzer":1}\n')
                print("BUZZER ON 신호 전송")
            elif cmd == "4":
                esp_conn.sendall(b'{"buzzer":0}\n')
                print("BUZZER OFF 신호 전송")
            elif cmd == "5":
                print("서버를 종료합니다.")
                import os
                os._exit(0)
            else:
                print("잘못된 입력입니다.")
        else:
            print("ESP8266가 연결되어 있지 않습니다.")

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"서버 실행 중... (TCP {HOST}:{PORT})")
    threading.Thread(target=cli, daemon=True).start()
    while True:
        print("연결 대기중");
        conn, addr = server.accept()
        print(f"[ESP8266] 연결됨: {addr}")
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    main()