/**
 * Static test for the actor network forward pass.
 * Compares C inference results against expected values from Python/ONNX.
 * 
 * Setup:
 *   1. Copy actor.h into this sketch folder
 *   2. Upload to ESP32
 *   3. Open Serial Monitor at 115200 baud
 */

#include <math.h>

#include "actor.h"

const float test_obs[ACTOR_OBS_SIZE] = {0.2,  1.0, -0.5,  0.5};

void setup() {
  unsigned long timestamp;
  float action[ACTOR_ACTION_SIZE];

  // Init serial (wait for connection)
  Serial.begin(115200);
  while(!Serial);
  Serial.println("Static inference test");

  // Get timestamp for measuring inference speed
  timestamp = micros();

  // Perform inference
  actor_forward(test_obs, action);

  // Print inference time and results
  Serial.print("Inference time (us): ");
  Serial.println(micros() - timestamp);
  Serial.print("Action: [");
  for (int i = 0; i < ACTOR_ACTION_SIZE; i++) {
    Serial.print(action[i], 4);
    if (i < ACTOR_ACTION_SIZE - 1) {
      Serial.print(", ");
    }
  }
  Serial.println("]");
}

void loop() {
  // Do nothing
}