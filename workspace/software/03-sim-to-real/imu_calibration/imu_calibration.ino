/**
 * One-time IMU calibration. Run this sketch once before deploying the balance
 * bot firmware. Calibration data is saved to NVS. Once done, run Examples >
 * M5Unified > Basic > Imu to see accel/gyro graphs.
 */

#include "M5Unified.h"

// Settings
static const uint8_t GYRO_CALIB_STR = 100; // Gyro calibration strength (0=disabled, 255=strongest)
static const uint32_t GYRO_CALIB_SEC = 10; // How long to hold still for gyro calibration
static const uint8_t ACCEL_CALIB_STR = 100; // Accel calibration strenth
static const uint32_t ACCEL_CALIB_SEC = 3; // How long to hold each face down
static const uint8_t NUM_FACES = 6; // Number of faces
static const String FACES[] = {"top",
                               "front",
                               "left",
                               "back",
                               "right",
                               "bottom"};

void setup() {
  float offset;

  // Initialize M5 library/drivers
  auto cfg = M5.config();
  M5.begin(cfg);

  // Initialize serial
  Serial.begin(115200);
  while (!Serial);
  Serial.println("\n=== IMU Calibration ===");

  // Print the IMU type
  Serial.printf("IMU type: ");
  switch (M5.Imu.getType()) {
      case m5::imu_mpu6886: 
        Serial.println("MPU6886"); 
        break;
      case m5::imu_bmi270:  
        Serial.println("BMI270");  
        break;
      case m5::imu_mpu6050: 
        Serial.println("MPU6050"); 
        break;
      case m5::imu_sh200q:  
        Serial.println("SH200Q");  
        break;
      case m5::imu_none:
      default:
        Serial.println("ERROR: IMU not found");
        while (1) {
          delay(1);
        }
  }

  // Check if calibration already exists
  if (M5.Imu.loadOffsetFromNVS()) {
    Serial.println("Existing calibration found in NVS.");
    Serial.println("WARNING: continuing will overwrite existing calibration data!");
    delay(2000);
  }

  // Print instructions and wait for 'enter' to continue
  Serial.println("Press 'enter' to start calibration process");
  while(Serial.read() != '\n');

  // Start gyroscope calibration
  Serial.println("\nStep 1: Gyro calibration");
  Serial.println("Place the robot on a flat, stable surface and don't move it.");
  Serial.println("Starting in 3 seconds...");
  delay(3000);

  // Perform gyroscope calibration (update every second)
  Serial.printf("Calibrating gyro for %d seconds...\n", GYRO_CALIB_SEC);
  M5.Imu.setCalibration(0, GYRO_CALIB_STR, 0);
  for (uint32_t i = 0; i < GYRO_CALIB_SEC; i++) {
      M5.Imu.update();
      delay(1000);
  }

  // Disable gyroscope calibration
  M5.Imu.setCalibration(0, 0, 0);
  Serial.println("Gyro calibration complete");

  // Start accelerometer calibration
  Serial.println("\nStep 2: Accelerometer calibration");
  Serial.println("As the name appears, slowly rotate the bot to place that face");
  Serial.println("on a flat, stable surface. Leave it there until the next face");
  Serial.println("name appears. Starting in 3 seconds...");
  delay(3000);

  // Perform accelerometer calibration
  M5.Imu.setCalibration(ACCEL_CALIB_STR, 0, 0);
  for (uint8_t i = 0; i < NUM_FACES; i++) {
    Serial.println(FACES[i]);
    for (uint32_t j = 0; j < ACCEL_CALIB_SEC; j++) {
      M5.Imu.update();
      delay(1000);
    }
  }

  // Disable accelerometer calibration
  M5.Imu.setCalibration(0, 0, 0);
  Serial.println("Accelometer calibration complete");

  // Save to NVS
  M5.Imu.saveOffsetToNVS();
  Serial.println("Calibration data saved to NVS");

  // Print offsets
  Serial.println("\nCalibration offsets:");
  for (int i = 0; i < 9; i++) {
    offset = M5.Imu.getOffsetData(i) * (1.0f / (1 << 19));
    Serial.printf("  offset[%d] = %.6f\n", i, offset);
  }
}

void loop() {
  // Do nothing
}
