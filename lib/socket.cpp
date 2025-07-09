// 이거 안됨

#include <ESP8266WiFi.h>
#include <WebSocketsClient.h>

// WiFi 설정
const char* ssid = "turtle";
const char* password = "turtlebot3";

// Flask 서버 WebSocket 주소 (예: ws://192.168.1.100:5000/ws)
const char* websocket_host = "192.168.0.67"; // Flask 서버 IP
const uint16_t websocket_port = 5000;
const char* websocket_path = "/AI_safety_monitoring";

// 하드웨어 핀 정의 (예시)
#define BUZZER_PIN D1    // 부저 = GPIO2
#define LED_PIN D2      // LED = GPIO14

WebSocketsClient webSocket;

// 현재 상태 변수
bool ledState = false;
bool buzzerState = false;

// 서버로 상태 전송 함수
void sendStatus() {
    String payload = "{\"led\":" + String(ledState ? 1 : 0) + ",\"buzzer\":" + String(buzzerState ? 1 : 0) + "}";
    webSocket.sendTXT(payload);
}

// WebSocket 이벤트 핸들러
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
    switch(type) {
        case WStype_CONNECTED:
            Serial.println("[AI_safety_monitoring] Connected to server");
            sendStatus(); // 연결되면 상태 전송
            break;
        case WStype_DISCONNECTED:
            Serial.println("[AI_safety_monitoring] Disconnected from server");
            break;
        case WStype_TEXT: {
            Serial.printf("[AI_safety_monitoring]AI_safety_monitoring Received: %s\n", payload);
            // 서버에서 {"led":0/1, "buzzer":0/1} 형태로 신호를 보낸다고 가정
            String msg = String((char*)payload);
            int led = msg.indexOf("\"led\":1") != -1 ? 1 : 0;
            int buzzer = msg.indexOf("\"buzzer\":1") != -1 ? 1 : 0;
            digitalWrite(LED_PIN, led);
            digitalWrite(BUZZER_PIN, buzzer);
            ledState = led;
            buzzerState = buzzer;
            // 상태 변경 시 서버에 다시 상태 전송
            sendStatus();
            break;
        }
        default:
            break;
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);

    // WiFi 연결
    WiFi.begin(ssid, password);
    Serial.print("WiFi connecting");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected!");

    // WebSocket 연결
    webSocket.begin(websocket_host, websocket_port, websocket_path);
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(5000); // 5초마다 재연결 시도
}

void loop() {
    webSocket.loop();

    // 예시: 10초마다 상태 서버로 전송
    static unsigned long lastSend = 0;
    if (millis() - lastSend > 10000) {
        sendStatus();
        lastSend = millis();
    }
}
