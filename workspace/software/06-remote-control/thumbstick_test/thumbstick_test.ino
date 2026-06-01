/**
 * AP WebSocket Thumbstick Test
 *
 * Core 0: WiFi AP + AsyncWebServer + WebSocket handler
 * Core 1: Receives thumbstick commands (printed to Serial for now)
 *
 * Dependencies (install via Arduino Library Manager):
 *   - ESP Async WebServer  (me-no-dev/ESPAsyncWebServer)
 *   - AsyncTCP             (me-no-dev/AsyncTCP)
 *
 * Once flashed:
 *   1. Connect to WiFi network "BalanceBot"  (password: "balancebot")
 *   2. Open browser to http://192.168.4.1
 *   3. Drag the thumbstick — watch Serial Monitor for values
 */

#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h>

#include "index_html.h"

// ---------------------------------------------------------------------------
// AP credentials and static IP
// ---------------------------------------------------------------------------
static const char* AP_SSID     = "BalanceBot";
static const char* AP_PASSWORD = "balancebot";

static const IPAddress AP_IP      (192, 168,  4,  1);
static const IPAddress AP_GATEWAY (192, 168,  4,  1);
static const IPAddress AP_SUBNET  (255, 255, 255,  0);

// ---------------------------------------------------------------------------
// Shared command state (written by Core 0 WS callback, read by Core 1)
// ---------------------------------------------------------------------------
static volatile float g_cmd_vel = 0.0f;  // forward/back  [-1, 1]
static volatile float g_cmd_yaw = 0.0f;  // turn rate     [-1, 1]
static portMUX_TYPE   g_cmd_mux = portMUX_INITIALIZER_UNLOCKED;

// Convenience helpers for safe read/write
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

// ---------------------------------------------------------------------------
// Server and WebSocket
// ---------------------------------------------------------------------------
AsyncWebServer server(80);
AsyncWebSocket ws("/ws");

// ---------------------------------------------------------------------------
// HTML + thumbstick UI
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// WebSocket event handler
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Server task — Core 0
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Arduino entry points — Core 1
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== AP WebSocket Thumbstick Test ===");

  xTaskCreatePinnedToCore(
    serverTask,
    "serverTask",
    8192,
    nullptr,
    1,
    nullptr,
    0    // Core 0
  );
}

void loop() {
  // Read shared commands and print them — replace this with
  // actor inference + motor control in the real firmware
  float vel, yaw;
  getCommands(vel, yaw);
  Serial.printf("[Core 1] cmd_vel=%.3f  cmd_yaw=%.3f\n", vel, yaw);
  vTaskDelay(pdMS_TO_TICKS(200));
}