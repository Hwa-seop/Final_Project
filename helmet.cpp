#include <ESP8266WiFi.h>

const char* ssid = "turtle";
const char* password = "turtlebot3";


// 서버 정보
const char* server_ip = "192.168.0.67"; // 서버 IP
const uint16_t server_port = 8000;

// 하드웨어 핀 정의
#define LED_PIN 4
#define BUZZER_PIN 5

WiFiClient client;

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

    // 서버 연결 시도
    if (client.connect(server_ip, server_port)) {
        Serial.println("서버에 TCP 연결 성공!");
        // 연결되면 상태 전송
        sendStatus();
    } else {
        Serial.println("서버에 TCP 연결 실패!");
    }
}

void sendStatus() {
    String payload = "{\"led\":" + String(digitalRead(LED_PIN)) + ",\"buzzer\":" + String(digitalRead(BUZZER_PIN)) + "}\n";
    client.print(payload);
}

void loop() {
    // 서버로부터 명령 수신
    if (client.connected() && client.available()) {
        String msg = client.readStringUntil('\n');
        Serial.print("서버로부터 수신: ");
        Serial.println(msg);

        int led = msg.indexOf("\"led\":1") != -1 ? 1 : 0;
        int buzzer = msg.indexOf("\"buzzer\":1") != -1 ? 1 : 0;
        digitalWrite(LED_PIN, led);
        digitalWrite(BUZZER_PIN, buzzer);

        // 상태 변경 시 서버에 다시 상태 전송
        sendStatus();
    }

    // 10초마다 상태 전송
    static unsigned long lastSend = 0;
    if (millis() - lastSend > 10000) {
        sendStatus();
        lastSend = millis();
    }
}