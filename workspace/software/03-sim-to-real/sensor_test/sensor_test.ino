#include <math.h>

#include "bala.h"
#include <M5Unified.h>

// Settings
const float TIMESTEP = 0.005f;          // Time (sec) between intervals
const float ALPHA = 0.99f;              // Alpha for complementary filter
const float ENC_TICKS_PER_REV = 420.0f; // Encoder ticks per wheel revolution

// Derived constants
const float ENC_TICKS_TO_RADS = (2.0f * M_PI) / ENC_TICKS_PER_REV;

// Globals
Bala bala;
float pitch = 0.0f;
int32_t prev_enc_left = 0;
int32_t prev_enc_right = 0;

void setup() {
  bool m5_ret;

  // Initialize M5 library/drivers
  auto cfg = M5.config();
  M5.begin(cfg);

  // Initialize serial
  Serial.begin(115200);
  while (!Serial);
  Serial.println("Sensor test");

  // Load calibration data from NVS
  m5_ret = M5.Imu.loadOffsetFromNVS();
  if (!m5_ret) {
    Serial.println("ERROR: No IMU calibration found!");
    Serial.println("Run calibrate_imu.ino first");
    while (1) {
      delay(1);
    }
  }

  // Zero out encoders
  bala.ClearEncoder();
}

void loop() {
  int32_t enc_left;
  int32_t enc_right;
  static int print_counter = 0;

  // Get timestamp for pacing to timestep interval
  unsigned long step_start = micros();

  // Read IMU
  M5.Imu.update();
  auto imu_data = M5.Imu.getImuData();s

  float accel_fwd  = imu_data.accel.y;
  float accel_up   = imu_data.accel.z;
  float pitch_rate = -imu_data.gyro.x * (M_PI / 180.0f);

  float accel_pitch = -atan2f(accel_fwd, accel_up);
  pitch = ALPHA * (pitch + (pitch_rate * TIMESTEP)) + ((1.0f - ALPHA) * accel_pitch);

  // Read encoders
  bala.GetEncoder(&enc_left, &enc_right);
  int32_t delta_enc_left  = enc_left  - prev_enc_left;
  int32_t delta_enc_right = enc_right - prev_enc_right;
  prev_enc_left  = enc_left;
  prev_enc_right = enc_right;

  float wheel_vel_left  = (float)delta_enc_left  * ENC_TICKS_TO_RADS / TIMESTEP;
  float wheel_vel_right = (float)delta_enc_right * ENC_TICKS_TO_RADS / TIMESTEP;

  // Pace to TIMESTEP before printing
  while (micros() - step_start < (unsigned long)(TIMESTEP * 1e6f));

  // Print every 20 iterations (~100ms) so Serial doesn't affect timing
  if (++print_counter >= 20) {
    print_counter = 0;
    Serial.printf("pitch=%.4f  pitch_rate=%.4f  vel_L=%.4f  vel_R=%.4f rad/s\n",
                  pitch, pitch_rate, wheel_vel_left, wheel_vel_right);
  }
}