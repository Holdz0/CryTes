#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include "DHT.h"
#include <Servo.h>

#define DHTPIN 2
#define DHTTYPE DHT11
#define BUZZER_PIN 8
#define SERVO_PIN 7

// Trafik ışıkları
#define GREEN_LED 10
#define YELLOW_LED 11
#define RED_LED 12

// Ninni LED'leri (mavi, sarı, yeşil)
int ninniLedler[] = {3, 4, 5};

#define RE  294
#define MI  329
#define FA  349
#define SOL 392
#define LA  440

int melodi[] = {
  LA, LA, SOL, LA, LA, SOL, FA, SOL, LA,
  FA, FA, MI, RE, MI, FA, MI, RE,
  LA, LA, SOL, LA, LA, SOL, FA, SOL, LA,
  FA, FA, MI, RE, MI, FA, MI, RE
};

int sureler[] = {
  4, 4, 2, 4, 4, 2, 4, 4, 2,
  4, 4, 2, 4, 2, 4, 4, 2,
  4, 4, 2, 4, 4, 2, 4, 4, 2,
  4, 4, 2, 4, 2, 4, 4, 1
};

LiquidCrystal_I2C lcd(0x27, 16, 2);
DHT dht(DHTPIN, DHTTYPE);
Servo oyuncakServo;

String inputString = "";
bool stringComplete = false;
float currentTemp = 0;
float currentHum = 0;

void setup() {
  Serial.begin(9600);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(YELLOW_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  
  for (int i = 0; i < 3; i++) {
    pinMode(ninniLedler[i], OUTPUT);
  }
  
  oyuncakServo.attach(SERVO_PIN);
  oyuncakServo.write(90);
  
  lcd.init();
  lcd.backlight();
  dht.begin();
  lcd.print("Bebek Analizi");
  lcd.setCursor(0, 1);
  lcd.print("Baslatiliyor...");
  setTrafficLight(0);
  delay(2000);
}

void playAlarm() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(200);
    digitalWrite(BUZZER_PIN, LOW);
    delay(100);
  }
}

void readSensor() {
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  if (!isnan(h) && !isnan(t)) {
    currentTemp = t;
    currentHum = h;
  }
}

void setTrafficLight(int state) {
  digitalWrite(GREEN_LED, state == 0 ? HIGH : LOW);
  digitalWrite(YELLOW_LED, state == 1 ? HIGH : LOW);
  digitalWrite(RED_LED, state == 2 ? HIGH : LOW);
}

void playLullaby() {
  int toplamNota = sizeof(melodi) / sizeof(melodi[0]);
  for (int i = 0; i < toplamNota; i++) {
    int notaSuresi = 1400 / sureler[i];
    digitalWrite(ninniLedler[i % 3], HIGH);
    tone(BUZZER_PIN, melodi[i], notaSuresi);
    delay(notaSuresi * 1.30);
    noTone(BUZZER_PIN);
    digitalWrite(ninniLedler[i % 3], LOW);
  }
  for (int i = 0; i < 3; i++) {
    digitalWrite(ninniLedler[i], LOW);
  }
}

void playToy() {
  for (int cycle = 0; cycle < 20; cycle++) {
    oyuncakServo.write(45);
    delay(500);
    oyuncakServo.write(135);
    delay(500);
  }
  oyuncakServo.write(90);
}

// 🎵🧸 NİNNİ + OYUNCAK BİRLİKTE
void playSoothe() {
  int toplamNota = sizeof(melodi) / sizeof(melodi[0]);
  
  for (int i = 0; i < toplamNota; i++) {
    int notaSuresi = 1400 / sureler[i];
    
    // LED yak
    digitalWrite(ninniLedler[i % 3], HIGH);
    
    // Servo hareket (her notada bir yöne)
    if (i % 2 == 0) {
      oyuncakServo.write(45);
    } else {
      oyuncakServo.write(135);
    }
    
    // Nota çal
    tone(BUZZER_PIN, melodi[i], notaSuresi);
    delay(notaSuresi * 1.30);
    noTone(BUZZER_PIN);
    
    // LED kapat
    digitalWrite(ninniLedler[i % 3], LOW);
  }
  
  // Bitince ortaya dön ve LED'leri kapat
  oyuncakServo.write(90);
  for (int i = 0; i < 3; i++) {
    digitalWrite(ninniLedler[i], LOW);
  }
}

void loop() {
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
  
  if (stringComplete) {
    inputString.trim();
    
    if (inputString == "LIGHT:GREEN") {
      setTrafficLight(0);
    } 
    else if (inputString == "LIGHT:YELLOW") {
      setTrafficLight(1);
    } 
    else if (inputString == "LIGHT:RED") {
      setTrafficLight(2);
    }
    else if (inputString == "PLAY_LULLABY") {
      playLullaby();
    }
    else if (inputString == "PLAY_TOY") {
      playToy();
    }
    else if (inputString == "PLAY_SOOTHE") {
      playSoothe();  // Ninni + Oyuncak birlikte
    }
    else if (inputString == "GET_SENSOR") {
      readSensor();
      Serial.print("SENSOR:");
      Serial.print(currentTemp, 1);
      Serial.print(",");
      Serial.println(currentHum, 1);
    } 
    else {
      int sep = inputString.indexOf('%');
      lcd.clear();
      if (sep != -1) {
        lcd.setCursor(0, 0);
        lcd.print(inputString.substring(0, sep).substring(0, 16));
        lcd.setCursor(0, 1);
        lcd.print(inputString.substring(sep + 1).substring(0, 16));
        if (inputString.indexOf("Guven") != -1) {
          playAlarm();
        }
      } else {
        lcd.print(inputString.substring(0, 16));
      }
    }
    inputString = "";
    stringComplete = false;
  }
}