#include <Arduino.h>

#define BUZZER_PIN D1    // 부저 = GPIO2
#define LED_PIN D2      // LED = GPIO14

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  // 켜기
  digitalWrite(LED_PIN, LOW);
  tone(BUZZER_PIN, 1000);
  delay(3000);  // 3초 대기

  // 끄기
  digitalWrite(LED_PIN, HIGH);
  noTone(BUZZER_PIN);
  delay(3000);  // 3초 대기
}
