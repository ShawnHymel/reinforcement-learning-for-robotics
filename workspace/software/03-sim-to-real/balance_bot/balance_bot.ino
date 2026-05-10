#include <math.h>
#include <M5Unified.h>

#include "actor.h"
#include "bala.h"

// Settings
#define DEBUG 0                         // Enable debug printing on intervals (can affect motion!)
const float TIMESTEP = 0.005f;          // Time (sec) between intervals
const float ALPHA = 0.99f;              // Alpha for complementary filter
const float MOTOR_SCALE = 1023.0f;      // Scale motors from [-1, 1] to [-1023, 1023]
const float ENC_TICKS_PER_REV = 420.0f; // Encoder ticks per wheel revolution
const float TIP_THRESHOLD = 0.79f;      // radians (~45 deg), stop motors if exceeded
const int16_t MOTOR_DIR_LEFT = -1;      // Left motor direction
const int16_t MOTOR_DIR_RIGHT = -1;     // Right motor direction

// Derived constants
const float ENC_TICKS_TO_RADS = (2.0f * M_PI) / ENC_TICKS_PER_REV;

// Globals
Bala bala;
float pitch = 0.0f;
int32_t prev_enc_left = 0;
int32_t prev_enc_right = 0;
bool tipped = false;

void setup() {
  bool m5_ret;

  // Initialize M5 library/drivers
  auto cfg = M5.config();
  M5.begin(cfg);

  // Initialize serial
  Serial.begin(115200);
  Serial.println("Balance bot");

  // Load calibration data from NVS
  m5_ret = M5.Imu.loadOffsetFromNVS();
  if (!m5_ret) {
    Serial.println("ERROR: No IMU calibration found!");
    Serial.println("Run calibrate_imu.ino first");
    while (1) {
      delay(1);
    }
  }
  Serial.println("IMU calibration loaded");

  // Zero out encoders
  bala.ClearEncoder();

  // Stop motors
  bala.SetSpeed(0, 0);

  // Wait to start
  Serial.println("Starting in 2 seconds, place robot upright");
  delay(2000);
  Serial.println("Running...");
}

void loop() {
  int32_t enc_left;
  int32_t enc_right;
  static int print_counter = 0;

  // Get timestamp for pacing to timestep interval
  unsigned long step_start = micros();

  // Read IMU
  M5.Imu.update();
  auto imu_data = M5.Imu.getImuData();
  float accel_fwd  = imu_data.accel.y;
  float accel_up   = imu_data.accel.z;

  // Use complementary filter to calculate pitch
  float pitch_rate = -imu_data.gyro.x * (M_PI / 180.0f);
  float accel_pitch = -atan2f(accel_fwd, accel_up);
  pitch = ALPHA * (pitch + (pitch_rate * TIMESTEP)) + ((1.0f - ALPHA) * accel_pitch);

  // Read encoders
  bala.GetEncoder(&enc_left, &enc_right);

  // Compute wheel velocities
  int32_t delta_enc_left  = enc_left  - prev_enc_left;
  int32_t delta_enc_right = enc_right - prev_enc_right;
  prev_enc_left  = enc_left;
  prev_enc_right = enc_right;
  float wheel_vel_left  = (float)delta_enc_left  * ENC_TICKS_TO_RADS / TIMESTEP;
  float wheel_vel_right = (float)delta_enc_right * ENC_TICKS_TO_RADS / TIMESTEP;

  // Check if tipped
  if (fabsf(pitch) > TIP_THRESHOLD) {
    tipped = true;
  }

  // Try to balance if not tipped
  if (!tipped) {
    // Build observation vector (much match training order):
    // [pitch, pitch_rate, wheel_vel_left, wheel_vel_right]
    float obs[ACTOR_OBS_SIZE] = {
      pitch,
      pitch_rate,
      wheel_vel_left,
      wheel_vel_right
    };

    // Run inference (actor network forward pass)
    float action[ACTOR_ACTION_SIZE];
    actor_forward(obs, action);

    // Clamp actions to [-1, 1] and scale to motor speed
    action[0] = constrain(action[0], -1.0f, 1.0f);
    action[1] = constrain(action[1], -1.0f, 1.0f);
    int16_t motor_left = MOTOR_DIR_LEFT * (int16_t)(action[0] * MOTOR_SCALE);
    int16_t motor_right = MOTOR_DIR_RIGHT * (int16_t)(action[1] * MOTOR_SCALE);

    // Set motor speed based on inference results
    bala.SetSpeed(motor_left, motor_right);
  
  // If tipped, shut off motors
  } else {
    bala.SetSpeed(0, 0);
  }

  // Pace to TIMESTEP before printing
  while (micros() - step_start < (unsigned long)(TIMESTEP * 1e6f));

  // Print diagnostics every few iterations
#if DEBUG
  if (++print_counter >= 20) {
  print_counter = 0;
  Serial.printf("pitch=% .3f  rate=% .3f  vL=% .2f  vR=% .2f  tipped=%d\n",
                pitch, pitch_rate, wheel_vel_left, wheel_vel_right, (int)tipped);
  }
#endif
}