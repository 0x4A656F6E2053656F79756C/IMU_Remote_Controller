#include <Wire.h>

#define MPU_ADDR 0x6A
#define DRDY_PIN 2

volatile bool interruptFlag = false;
volatile bool isRunning = false; // 전송 상태 제어 플래그

int16_t ax, ay, az;
int16_t gx, gy, gz;

void onDataReady() {
  interruptFlag = true;
}

void writeRegister(uint8_t reg, uint8_t data) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(data);
  Wire.endTransmission();
}

uint8_t readRegister(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 1);

  if (Wire.available()) return Wire.read();
  return 0;
}

void setup() {
  Serial.begin(500000); // 고속 통신
  Wire.begin();
  Wire.setClock(400000); // I2C Fast Mode

  pinMode(DRDY_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(DRDY_PIN), onDataReady, RISING);

  // 소프트웨어 리셋 트리거
  writeRegister(0x12, 0x01);
  delay(50); 

  // accel 0x70 for ~833 Hz
  writeRegister(0x10, 0x70);
  // gyro 0x70 for ~833 Hz
  writeRegister(0x11, 0x70);
  // accel + gyro DRDY 둘 다 사용 설정
  writeRegister(0x0D, 0x03);

  delay(100);
}

void loop() {
  // 1. 키보드 입력 체크 ('s' 입력 시 시작/중지)
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 's' || c == 'S') {
      isRunning = !isRunning;
    }
  }

  // 중지 상태이거나 데이터가 준비되지 않았으면 리턴
  if (!isRunning || !interruptFlag) return;
  interruptFlag = false;

  // STATUS_REG 확인 (Accel & Gyro 데이터 모두 준비될 때까지 대기)
  uint8_t status = readRegister(0x1E);
  while (!((status & 0x01) && (status & 0x02))) {
    status = readRegister(0x1E);
  }

  // 데이터 읽기 (0x22번지부터 12바이트: Gyro 6바이트 + Accel 6바이트)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x22);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 12);

  if (Wire.available() == 12) {
    gx = Wire.read() | (Wire.read() << 8);
    gy = Wire.read() | (Wire.read() << 8);
    gz = Wire.read() | (Wire.read() << 8);

    ax = Wire.read() | (Wire.read() << 8);
    ay = Wire.read() | (Wire.read() << 8);
    az = Wire.read() | (Wire.read() << 8);

    uint32_t t = micros();

    // 바이너리 패킷 전송 (총 18바이트)
    Serial.write(0xAA); // Header 1
    Serial.write(0x55); // Header 2
    Serial.write((uint8_t*)&t, 4);   // Timestamp (4 bytes)
    Serial.write((uint8_t*)&gx, 2);  // Gyro X (2 bytes)
    Serial.write((uint8_t*)&gy, 2);  // Gyro Y (2 bytes)
    Serial.write((uint8_t*)&gz, 2);  // Gyro Z (2 bytes)
    Serial.write((uint8_t*)&ax, 2);  // Accel X (2 bytes)
    Serial.write((uint8_t*)&ay, 2);  // Accel Y (2 bytes)
    Serial.write((uint8_t*)&az, 2);  // Accel Z (2 bytes)
  }
}