#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <ArduinoJson.h>

// WiFi 설정
const char* ssid = "turtle";
const char* password = "turtlebot3";

// 서버 정보
const char* server_url = "http://192.168.0.74:5000/control"; // Flask HTTP 라우트 주소

// 하드웨어 핀 정의
#define LED_PIN D2
#define BUZZER_PIN D1

void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);

    WiFi.begin(ssid, password);
    Serial.print("WiFi connecting");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected!");
}

void sendStatusAndReceiveCommand() {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;

        http.begin(server_url);
        http.addHeader("Content-Type", "application/json");

        // 상태 JSON 만들기
        String payload = "{\"led\":" + String(digitalRead(LED_PIN)) + ",\"buzzer\":" + String(digitalRead(BUZZER_PIN)) + "}";

        int httpCode = http.POST(payload);

        if (httpCode > 0) {
            String response = http.getString();
            Serial.print("서버 응답: ");
            Serial.println(response);

            // JSON 파싱
            DynamicJsonDocument doc(256);
            DeserializationError error = deserializeJson(doc, response);

            if (!error) {
                int led = doc["led"];
                int buzzer = doc["buzzer"];

                digitalWrite(LED_PIN, led ? LOW : HIGH);  // 1이면 꺼짐
                if (buzzer == 1) {
                    tone(BUZZER_PIN, 1000);
                } else {
                    noTone(BUZZER_PIN);
                }
            } else {
                Serial.println("응답 JSON 파싱 실패");
            }
        } else {
            Serial.print("HTTP 요청 실패, 코드: ");
            Serial.println(httpCode);
        }

        http.end();
    } else {
        Serial.println("WiFi 연결 안 됨, 재시도 중...");
    }
}

void loop() {
    sendStatusAndReceiveCommand();
    delay(10000); // 10초 간격 요청
}
