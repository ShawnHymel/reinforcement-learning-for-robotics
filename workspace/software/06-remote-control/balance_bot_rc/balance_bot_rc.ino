#include <math.h>
#include <M5Unified.h>
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h>

#include "actor.h"
#include "bala.h"
#include "index_html.h"

// Enable debug printing on intervals (can affect motion!)
#define DEBUG 0                         

// WiFi settings
static const char* AP_SSID     = "BalanceBot";
static const char* AP_PASSWORD = "balancebot";
static const IPAddress AP_IP      (192, 168, 4, 1);
static const IPAddress AP_GATEWAY (192, 168, 4, 1);
static const IPAddress AP_SUBNET  (255, 255, 255, 0);

// Robot settings
const float VEL_FACTOR = 0.5f;          // Max velocity that was used in training (negative to flip direction)
const float YAW_FACTOR = -1.5f;         // Max yaw that was used in training (negative to flip direction)
const float CMD_DEADZONE = 0.0f;        // Thumbstick deadzone (ignore inputs below this value)
const float PITCH_OFFSET = -0.05f;        // Tune this so the robot stays upright (+: back bias, -: front bias)
const float MOTOR_BOOST = 1.0f;         // Tune this so the motors are responsive on battery power
const float ACTION_DEADBAND = 0.0f;     // Tune this: Ignore small motor corrections
const float ACTION_ALPHA = 0.0f;        // Tune this: alpha for low-pass filter (higher: smoother, more lag)
const float COMP_ALPHA = 0.99f;         // Alpha for complementary filter (must match training)
const float TIMESTEP = 0.005f;          // Time (sec) between intervals
const float MOTOR_SCALE = 1023.0f;      // Scale motors from [-1, 1] to [-1023, 1023]
const float ENC_TICKS_PER_REV = 420.0f; // Encoder ticks per wheel revolution
const float TIP_THRESHOLD = 0.79f;      // radians (~45 deg), stop motors if exceeded
const int16_t MOTOR_DIR_LEFT = -1;      // Left motor direction
const int16_t MOTOR_DIR_RIGHT = -1;     // Right motor direction
const int32_t ENC_DIR_LEFT = 1;         // Left encoder direction
const int32_t ENC_DIR_RIGHT = 1;        // Right encoder direction
const unsigned long RESET_TIME_MS = 1000; // How long to wait before running again

// Derived constants
const float ENC_TICKS_TO_RADS = (2.0f * M_PI) / ENC_TICKS_PER_REV;

// Globals
Bala bala;
float pitch = 0.0f;
int32_t prev_enc_left = 0;
int32_t prev_enc_right = 0;
bool tipped = false;
float action_filtered[2] = {0.0f, 0.0f};
static volatile float g_cmd_vel = 0.0f;
static volatile float g_cmd_yaw = 0.0f;
static portMUX_TYPE g_cmd_mux = portMUX_INITIALIZER_UNLOCKED;

// Webserver and socket
AsyncWebServer server(80);
AsyncWebSocket ws("/ws");

//******************************************************************************
// Functions

inline void setCommands(float vel, float yaw) {
  portENTER_CRITICAL(&g_cmd_mux);
  g_cmd_vel = vel;
  g_cmd_yaw = yaw;
  portEXIT_CRITICAL(&g_cmd_mux);
}

inline void getCommands(float& vel, float& yaw) {
  portENTER_CRITICAL(&g_cmd_mux);
  vel = g_cmd_vel;
  yaw = g_cmd_yaw;
  portEXIT_CRITICAL(&g_cmd_mux);
}

// WebSocket handler (Core 0)
void onWsEvent(AsyncWebSocket* server,
               AsyncWebSocketClient* client,
               AwsEventType type,
               void* arg,
               uint8_t* data,
               size_t len)
{
  if (type == WS_EVT_CONNECT) {
    Serial.printf("[WS] Client #%u connected\n", client->id());

  } else if (type == WS_EVT_DISCONNECT) {
    Serial.printf("[WS] Client #%u disconnected — zeroing commands\n", client->id());
    setCommands(0.0f, 0.0f);

  } else if (type == WS_EVT_DATA) {
    AwsFrameInfo* info = (AwsFrameInfo*)arg;
    // Only handle complete, text frames
    if (info->final && info->index == 0 && info->len == len
        && info->opcode == WS_TEXT) {
      // Null-terminate and parse JSON
      char buf[64];
      size_t copy_len = len < sizeof(buf) - 1 ? len : sizeof(buf) - 1;
      memcpy(buf, data, copy_len);
      buf[copy_len] = '\0';

      StaticJsonDocument<64> doc;
      if (deserializeJson(doc, buf) == DeserializationError::Ok) {
        float vel = doc["vel"] | 0.0f;
        float yaw = doc["yaw"] | 0.0f;

        // Clamp just in case
        vel = constrain(vel, -1.0f, 1.0f);
        yaw = constrain(yaw, -1.0f, 1.0f);

        setCommands(vel, yaw);
      }
    }
  }
}

//******************************************************************************
// Server thread (Core 0)

void serverTask(void* pvParameters) {
  Serial.println("[Core 0] Starting WiFi AP...");

  WiFi.softAPConfig(AP_IP, AP_GATEWAY, AP_SUBNET);
  WiFi.softAP(AP_SSID, AP_PASSWORD);

  Serial.printf("[Core 0] AP started: SSID=%s  IP=%s\n",
                AP_SSID,
                WiFi.softAPIP().toString().c_str());

  ws.onEvent(onWsEvent);
  server.addHandler(&ws);

  server.on("/", HTTP_GET, [](AsyncWebServerRequest* request) {
    request->send_P(200, "text/html", INDEX_HTML);
  });
  server.onNotFound([](AsyncWebServerRequest* request) {
    request->send(404, "text/plain", "Not found");
  });

  server.begin();
  Serial.println("[Core 0] HTTP + WebSocket server running");

  // Periodically clean up stale WS clients
  for (;;) {
    ws.cleanupClients();
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}

//******************************************************************************
// Main (Core 1)

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

  // Start WebServer thread on Core 0
  xTaskCreatePinnedToCore(
    serverTask,
    "serverTask",
    8192,
    nullptr,
    1,
    nullptr,
    0    // Core 0
  );

  // Wait to start
  Serial.printf("Starting in %lu seconds, place robot upright\n", RESET_TIME_MS / 1000);
  delay(RESET_TIME_MS);
  Serial.println("Running...");
}

void loop() {
  int32_t enc_left;
  int32_t enc_right;
  static int print_counter = 0;
  float cmd_vel = 0.0f;
  float cmd_yaw = 0.0f;

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
  pitch = COMP_ALPHA * (pitch + (pitch_rate * TIMESTEP)) + ((1.0f - COMP_ALPHA) * accel_pitch);

  // Read encoders (flip sign if needed)
  bala.GetEncoder(&enc_left, &enc_right);
  enc_left = ENC_DIR_LEFT * enc_left;
  enc_right = ENC_DIR_RIGHT * enc_right;

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
    // Get last command from WebSocket
    getCommands(cmd_vel, cmd_yaw);

    // Apply deadzone
    cmd_vel = fabsf(cmd_vel) < CMD_DEADZONE ? 0.0f : cmd_vel;
    cmd_yaw = fabsf(cmd_yaw) < CMD_DEADZONE ? 0.0f : cmd_yaw;

    Serial.print(cmd_vel * VEL_FACTOR);
    Serial.print(",");
    Serial.println(cmd_yaw * YAW_FACTOR);

    // Build observation vector (much match training order):
    // [pitch, pitch_rate, wheel_vel_left, wheel_vel_right, cmd_vel, cmd_yaw]
    float obs[ACTOR_OBS_SIZE] = {
      pitch + PITCH_OFFSET,
      pitch_rate,
      wheel_vel_left,
      wheel_vel_right,
      cmd_vel * VEL_FACTOR,
      cmd_yaw * YAW_FACTOR
    };

    // Run inference (actor network forward pass)
    float action[ACTOR_ACTION_SIZE];
    actor_forward(obs, action);

    // Clamp actions to [-1, 1]
    action[0] = constrain(action[0], -1.0f, 1.0f);
    action[1] = constrain(action[1], -1.0f, 1.0f);

    // Use a low-pass filter to smooth out the motor commands
    action_filtered[0] = (ACTION_ALPHA * action_filtered[0]) + 
                         ((1.0f - ACTION_ALPHA) * action[0]);
    action_filtered[1] = (ACTION_ALPHA * action_filtered[1]) +
                         ((1.0f - ACTION_ALPHA) * action[1]);

    // Apply action deadband to prevent small movements
    float action_motors[2];
    action_motors[0] = (fabs(action_filtered[0]) < ACTION_DEADBAND) ? 0.0f : action_filtered[0];
    action_motors[1] = (fabs(action_filtered[1]) < ACTION_DEADBAND) ? 0.0f : action_filtered[1];

    // Calculate actual motor values from normalized values (boost if needed)
    int16_t motor_left = MOTOR_DIR_LEFT * 
                          (int16_t)(action_motors[0] * MOTOR_SCALE * MOTOR_BOOST);
    int16_t motor_right = MOTOR_DIR_RIGHT * 
                          (int16_t)(action_motors[1] * MOTOR_SCALE* MOTOR_BOOST);

    // Set motor speed based on inference results
    bala.SetSpeed(motor_left, motor_right);
  
  // If tipped, shut off motors and wait to be turned upright
  } else {
    bala.SetSpeed(0, 0);
    // Wait for someone to pick the bot up
    if (fabsf(pitch) <= 0.3) {
      tipped = false;

      // Reset the tip sensor pitch and filters
      pitch = 0.0f;
      action_filtered[0] = 0.0f;
      action_filtered[1] = 0.0f;

      // Reset the encoder counters
      bala.ClearEncoder();
      prev_enc_left = 0;
      prev_enc_right = 0;

      // Wait a moment before starting
      Serial.printf("Untipped! Starting in %lu seconds\n", RESET_TIME_MS / 1000);
      delay(RESET_TIME_MS);
      Serial.printf("Running...");
    }
  }

  // Pace to TIMESTEP before printing
  // Serial.println(micros() - step_start);
  while ((micros() - step_start) < (unsigned long)(TIMESTEP * 1e6f));

  // Print diagnostics every few iterations
#if DEBUG
  if (++print_counter >= 20) {
  print_counter = 0;
  int32_t batt_lvl = M5.Power.getBatteryLevel();
  int16_t batt_mv = M5.Power.getBatteryVoltage();
  Serial.printf("batt_lvl=%d batt_mv=%d pitch=%.3f rate=%.3f vL=%.2f vR=%.2f tipped=%d cmd_vel=%.2f cmd_yaw=%.2f\n",
                batt_lvl, 
                batt_mv, 
                pitch, 
                pitch_rate, 
                wheel_vel_left, 
                wheel_vel_right, 
                (int)tipped);
  }
#endif
}