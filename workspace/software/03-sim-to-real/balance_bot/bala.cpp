#include "bala.h"
#include <Wire.h>

// BALA2 STM32 I2C address and registers
static const uint8_t BALA2_ADDR = 0x3A;
static const uint8_t REG_ENCODER = 0x10;
static const uint8_t REG_MOTOR = 0x00;
static const uint8_t REG_SERVO = 0x20;
static const uint8_t REG_SERVO_PWM = 0x30;

// ---------------------------------------------------------------------------
// Internal I2C helpers

static void i2c_write(uint8_t reg, const uint8_t* data, size_t len) {
    Wire.beginTransmission(BALA2_ADDR);
    Wire.write(reg);
    Wire.write(data, len);
    Wire.endTransmission();
}

static void i2c_read(uint8_t reg, uint8_t* data, size_t len) {
    Wire.beginTransmission(BALA2_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);  // repeated start
    Wire.requestFrom(BALA2_ADDR, (uint8_t)len);
    for (size_t i = 0; i < len; i++) {
        data[i] = Wire.available() ? Wire.read() : 0;
    }
}

// ---------------------------------------------------------------------------
// Bala class

Bala::Bala() {
  wheel_left_encoder = 0;
  wheel_right_encoder = 0;
  i2c_mutex = NULL;
}

void Bala::ClearEncoder() {
  SetEncoder(0, 0);
  wheel_left_encoder = 0;
  wheel_right_encoder = 0;
}

void Bala::GetEncoder(int32_t* wheel_left, int32_t* wheel_right) {
  UpdateEncoder();
  *wheel_left = wheel_left_encoder;
  *wheel_right = wheel_right_encoder;
}

void Bala::SetEncoder(int32_t wheel_left, int32_t wheel_right) {
  uint8_t data_out[8];
  data_out[0] = (uint8_t)(wheel_left >> 24);
  data_out[1] = (uint8_t)(wheel_left >> 16);
  data_out[2] = (uint8_t)(wheel_left >> 8);
  data_out[3] = (uint8_t)(wheel_left >> 0);

  data_out[4] = (uint8_t)(wheel_right >> 24);
  data_out[5] = (uint8_t)(wheel_right >> 16);
  data_out[6] = (uint8_t)(wheel_right >> 8);
  data_out[7] = (uint8_t)(wheel_right >> 0);
  if(i2c_mutex != NULL) { xSemaphoreTake(i2c_mutex, portMAX_DELAY); }
  i2c_write(REG_ENCODER, data_out, 8);
  if(i2c_mutex != NULL) { xSemaphoreGive(i2c_mutex); }
}

void Bala::UpdateEncoder() {
  uint8_t data_in[8];

  if(i2c_mutex != NULL) { xSemaphoreTake(i2c_mutex, portMAX_DELAY); }
  i2c_read(REG_ENCODER, data_in, 8);
  if(i2c_mutex != NULL) { xSemaphoreGive(i2c_mutex); }

  wheel_left_encoder = (data_in[0] << 24) | (data_in[1] << 16) | (data_in[2] << 8) | data_in[3];
  wheel_right_encoder = (data_in[4] << 24) | (data_in[5] << 16) | (data_in[6] << 8) | data_in[7]; 
}

void Bala::SetSpeed(int16_t wheel_left, int16_t wheel_right) {
  uint8_t data_out[4];

  data_out[0] = (int8_t)(wheel_left >> 8);
  data_out[1] = (int8_t)(wheel_left >> 0);
  data_out[2] = (int8_t)(wheel_right >> 8);
  data_out[3] = (int8_t)(wheel_right >> 0);

  if(i2c_mutex != NULL) { xSemaphoreTake(i2c_mutex, portMAX_DELAY); }
  i2c_write(REG_MOTOR, data_out, 4);
  if(i2c_mutex != NULL) { xSemaphoreGive(i2c_mutex); }
}

void Bala::SetMutex(SemaphoreHandle_t* mutex) {
  i2c_mutex = *mutex;
}

void Bala::SetServoAngle(uint8_t pos, uint8_t angle) {
  if (pos < 1) {
    pos = 1;
  } else if (pos > 8) {
    pos = 8;
  }

  pos = pos - 1;

  if(i2c_mutex != NULL) { xSemaphoreTake(i2c_mutex, portMAX_DELAY); }
  i2c_write(REG_SERVO | pos, &angle, 1);
  if(i2c_mutex != NULL) { xSemaphoreGive(i2c_mutex); }
}

void Bala::SetServoPulse(uint8_t pos, uint16_t width) {
  if (pos < 1) {
    pos = 1;
  } else if (pos > 8) {
    pos = 8;
  }

  pos = (pos - 1) << 1;
  uint8_t buff_out[2];
  buff_out[0] = width >> 8;
  buff_out[1] = width & 0xff;

  if(i2c_mutex != NULL) { xSemaphoreTake(i2c_mutex, portMAX_DELAY); }
  i2c_write(REG_SERVO_PWM | pos, buff_out, 2);
  if(i2c_mutex != NULL) { xSemaphoreGive(i2c_mutex); }
}

