#include <Wire.h>

#define MPU_ADDR 0x6A
#define DRDY_PIN 2

volatile bool interruptFlag = false;

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
  Serial.begin(921600);
  Wire.begin();
  Wire.setClock(400000);

  pinMode(DRDY_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(DRDY_PIN), onDataReady, RISING);

  // auto increment
  writeRegister(0x12, 0x01);

  // accel 0x70 for ~833 Hz
  writeRegister(0x10, 0x70);

  // gyro 0x70 for ~833 Hz
  writeRegister(0x11, 0x70);

  // accel + gyro DRDY 둘 다 사용
  writeRegister(0x0D, 0x03);

  delay(100);
}

void loop() {
  if (!interruptFlag) return;
  interruptFlag = false;

  // STATUS_REG 확인
  uint8_t status = readRegister(0x1E);

  bool accel_ready = status & 0x01;
  bool gyro_ready  = status & 0x02;

  // 둘 다 준비 안됐으면 무시
  if (!(accel_ready && gyro_ready)) return;

  // 여기서만 읽기 (중복 제거 핵심)
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

    Serial.write(0xAA);
    Serial.write(0x55);
    Serial.write((uint8_t*)&t, 4);

    Serial.write((uint8_t*)&gx, 2);
    Serial.write((uint8_t*)&gy, 2);
    Serial.write((uint8_t*)&gz, 2);

    Serial.write((uint8_t*)&ax, 2);
    Serial.write((uint8_t*)&ay, 2);
    Serial.write((uint8_t*)&az, 2);
  }
}