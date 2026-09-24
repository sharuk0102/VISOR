/*
 * ==================================================================================
 * PROJECT: Sipeed Maixduino AprilTag Relay System
 * MODULE: Internal ESP32 Sketch (Sender)
 * BOARD: ESP32 (Generic, Arduino core)
 * 
 * IMPORTANT BOARD INFO:
 *   This board is a Sipeed Maixduino. K210 and ESP32 are hardwired on the PCB via
 *   K210 IO6/IO7 <--> ESP32 GPIO1/GPIO3 (UART0). We use Serial2 (UART2) to avoid
 *   IO_MUX locks on UART0 pins.
 * 
 * WIRING SUMMARY (Hardwired on PCB):
 *   - RX pin = GPIO3 (receives from K210 IO7)
 *   - TX pin = GPIO1 (would transmit to K210 IO6, not actively used for this project)
 *   - USB Serial (CH552 Chip) is also shared on the same UART0 lines.
 * ==================================================================================
 */

#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_arduino_version.h>

// --- CONSTANTS & CONFIGURATION ────────────────────────────────────────────────────
const unsigned long UART_BAUD    = 115200; // Baud rate for K210 & USB connection
const int WIFI_CHANNEL           = 1;      // WiFi Channel (MUST match external ESP32)
const unsigned long HEARTBEAT_TIMEOUT_MS = 2000; // Keepalive heartbeat transmission interval (2 sec)

// Local debug flag.
// Set to 1 ONLY for local debugging when K210 is disconnected or you are diagnosing.
// Set to 0 for normal operation to avoid injecting debug bytes back onto the shared K210 RX line.
#define DEBUG_LOCAL 0

// Destination Peer MAC Address (External ESP32 Station MAC)
uint8_t peerMac[] = {0x28, 0x05, 0xA5, 0xE2, 0xC6, 0x64}; // Receiver ESP32 MAC: 28:05:A5:E2:C6:64

// --- DATA STRUCTURES ──────────────────────────────────────────────────────────────
typedef struct {
  bool detected;       // True if tag is detected, false otherwise
  int tagID;           // The ID of the detected tag (-1 if not detected)
  int centerX;         // X center coordinate (-1 if not detected/applicable)
  int centerY;         // Y center coordinate (-1 if not detected/applicable)
  uint32_t timestamp;  // Millisecond timestamp of transmission
} tag_message;

// --- GLOBAL VARIABLES ─────────────────────────────────────────────────────────────
char buffer[128];                    // Buffer for assembling K210 serial line
int bufIdx = 0;                      // Current write index in the buffer
unsigned long lastK210RxTime = 0;    // Timestamp of the last valid frame from the K210
volatile bool shouldPrintSentStatus = false; // State flag to avoid forwarding print spam

// Diagnostic variables for onboard LED pulsing on successful parse
unsigned long pulseOnTime = 0;
bool ledActive = false;

// --- ESP-NOW CORE VERSION COMPATIBILITY ───────────────────────────────────────────
#if defined(ESP_ARDUINO_VERSION) && (ESP_ARDUINO_VERSION >= ESP_ARDUINO_VERSION_VAL(3, 0, 0))
  #define ESP_NOW_CORE_V3
#endif

// ESP-NOW Send Callback
#if defined(ESP_NOW_CORE_V3)
void OnDataSent(const esp_now_send_info_t *info, esp_now_send_status_t status) {
#else
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
#endif
  // Report transmission status back to the K210 processor
  if (status == ESP_NOW_SEND_SUCCESS) {
    Serial2.println("SEND_OK");
  } else {
    Serial2.println("SEND_FAIL");
  }

  if (shouldPrintSentStatus) {
    if (status == ESP_NOW_SEND_SUCCESS) {
      Serial.println("[INTERNAL ESP] Sent successfully");
    } else {
      Serial.println("[INTERNAL ESP] Send FAILED");
    }
    shouldPrintSentStatus = false;
  }
}

// --- HELPER FUNCTIONS ─────────────────────────────────────────────────────────────

// Trim trailing/leading whitespace and carriage returns in-place
void trimInPlace(char *s) {
  int len = strlen(s);
  // Trim trailing whitespace
  while (len > 0 && (s[len - 1] == ' ' || s[len - 1] == '\t' || s[len - 1] == '\r' || s[len - 1] == '\n')) {
    s[--len] = '\0';
  }
  // Find index of first non-whitespace character
  int start = 0;
  while (s[start] != '\0' && (s[start] == ' ' || s[start] == '\t' || s[start] == '\r' || s[start] == '\n')) {
    start++;
  }
  // Shift string left to overwrite leading whitespace
  if (start > 0) {
    memmove(s, s + start, len - start + 1);
  }
}

// Packages the payload and transmits it over ESP-NOW
void sendToExternal(bool detected, int tagID, int centerX, int centerY) {
  tag_message msg;
  msg.detected = detected;
  msg.tagID = tagID;
  msg.centerX = centerX;
  msg.centerY = centerY;
  msg.timestamp = millis();

  // Print right before forwarding/sending (ONLY on state change)
  static bool lastDetectedVal = false;
  static int lastSentTagID = -2;
  static bool firstSendVal = true;

  if (firstSendVal || (detected != lastDetectedVal) || (detected && tagID != lastSentTagID)) {
    Serial.println("[INTERNAL ESP] Forwarding data to External ESP32...");
    lastDetectedVal = detected;
    lastSentTagID = tagID;
    firstSendVal = false;
    shouldPrintSentStatus = true;
  } else {
    shouldPrintSentStatus = false;
  }

  // Send packet
  esp_err_t result = esp_now_send(peerMac, (uint8_t *)&msg, sizeof(msg));
  if (result != ESP_OK && shouldPrintSentStatus) {
    Serial.println("[INTERNAL ESP] Forwarding failed immediately (esp_now_send error)");
  }
}

// Parses a complete trimmed ASCII line received from the K210
void parseK210Line(char* line) {
  #if DEBUG_LOCAL
  Serial.print("[K210 UART RX] ");
  Serial.println(line);
  #endif

  bool tagDetected = false;
  int tagID = -1;
  int centerX = -1;
  int centerY = -1;
  bool parsedSuccessfully = false;

  // 1. Check for tag-not-detected strings
  if (strstr(line, "Helmet Not Detected") != NULL || 
      strstr(line, "HELMET NOT DETECTED") != NULL ||
      strstr(line, "NO HELMET") != NULL ||
      strstr(line, "Not Detected") != NULL ||
      strstr(line, "TAG:0") != NULL) {
    tagDetected = false;
    tagID = -1;
    parsedSuccessfully = true;
  }
  // 2. Check for tag-detected strings
  else {
    char* tagPtr = strstr(line, "TAG:");
    if (tagPtr != NULL) {
      int tagVal = atoi(tagPtr + 4);
      if (tagVal == 1) {
        tagDetected = true;
        parsedSuccessfully = true;

        // Extract details: ID
        char* idPtr = strstr(line, "ID:");
        if (idPtr != NULL) {
          tagID = atoi(idPtr + 3);
        }

        // Extract details: X
        char* xPtr = strstr(line, "X:");
        if (xPtr != NULL) {
          centerX = atoi(xPtr + 2);
        }

        // Extract details: Y
        char* yPtr = strstr(line, "Y:");
        if (yPtr != NULL) {
          centerY = atoi(yPtr + 2);
        }
      }
    }
  }

  if (parsedSuccessfully) {
    // Reset heartbeat timer as we received a valid status from the K210
    lastK210RxTime = millis();

    // Pulse the onboard LED (GPIO 2) to diagnose real incoming data.
    // This helps distinguish internal Serial2 reception from CH552 hardware echo.
    // --- DISABLED PHYSICAL GPIO OUTPUTS AS PER REQUIREMENTS ---
    // digitalWrite(2, HIGH);
    // pulseOnTime = millis();
    // ledActive = true;

    // Print when data is RECEIVED from Maixduino (ONLY on state change)
    static bool lastDetectedState = false;
    static int lastTagID = -2;
    static bool firstStateUpdate = true;

    if (firstStateUpdate || (tagDetected != lastDetectedState) || (tagDetected && tagID != lastTagID)) {
      if (tagDetected) {
        Serial.printf("[INTERNAL ESP] Received AprilTag ID: %d from Maixduino\n", tagID);
      } else {
        Serial.println("[INTERNAL ESP] Received: No helmet from Maixduino");
      }
      lastDetectedState = tagDetected;
      lastTagID = tagID;
      firstStateUpdate = false;
    }

    // Send status to external ESP32
    sendToExternal(tagDetected, tagID, centerX, centerY);
  } else {
    #if DEBUG_LOCAL
    Serial.println("[Parser] Warning: Received line format unrecognized. Discarding line.");
    #endif
  }
}

// Reads serial data non-blockingly from K210 (UART2 mapped to GPIO1)
void readFromK210() {
  while (Serial2.available() > 0) {
    char c = Serial2.read();

    // If newline or carriage return is hit, process the line
    if (c == '\n' || c == '\r') {
      if (bufIdx > 0) {
        buffer[bufIdx] = '\0';
        
        // Trim in-place to handle leading/trailing spacing
        trimInPlace(buffer);
        if (strlen(buffer) > 0) {
          parseK210Line(buffer);
        }
        
        bufIdx = 0; // Reset buffer pointer
      }
    } else {
      // Add character to buffer if there is space
      if (bufIdx < sizeof(buffer) - 1) {
        buffer[bufIdx++] = c;
      } else {
        #if DEBUG_LOCAL
        Serial.println("[UART] Buffer overflow. Discarding line.");
        #endif
        bufIdx = 0;
      }
    }
  }
}

// --- MAIN FUNCTIONS ──────────────────────────────────────────────────────────────

void setup() {
  // Configure diagnostic onboard LED
  // pinMode(2, OUTPUT);
  // digitalWrite(2, LOW);

  // Note: On Maixduino, this is shared with the K210 and USB interface.
  // We initialize the default Serial in TX-only mode to prevent stealing GPIO 3 from Serial2.
  Serial.begin(UART_BAUD, SERIAL_8N1, -1, 1);
  delay(100);

  // Initialize Serial2 for K210 communication.
  // RX pin = GPIO3 (receives from K210 IO7)
  // TX pin = GPIO1 (would transmit to K210 IO6, not actively used for this project)
  Serial2.begin(UART_BAUD, SERIAL_8N1, 3, 1);
  
  Serial.println("[INTERNAL ESP] Connected. Ready to receive from Maixduino.");

  // Initialize WiFi in Station mode
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(); // Disconnect from any saved network profile

  // Start the WiFi interface (Required before setting the channel manually)
#if defined(ESP_NOW_CORE_V3)
  esp_wifi_start();
#endif

  // Force WiFi to operate on a fixed channel (MUST match external ESP32)
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);

  #if DEBUG_LOCAL
  Serial.printf("[WiFi] Operating on Channel %d.\n", WIFI_CHANNEL);
  #endif

  // Initialize ESP-NOW
  if (esp_now_init() != ESP_OK) {
    #if DEBUG_LOCAL
    Serial.println("[ESP-NOW] Fatal: Initialization failed.");
    #endif
    while (true) { delay(1000); } // Halt execution
  }

  // Register ESP-NOW Send Callback
  esp_now_register_send_cb(OnDataSent);

  // Configure Peer (External ESP32)
  esp_now_peer_info_t peerInfo;
  memset(&peerInfo, 0, sizeof(peerInfo));
  memcpy(peerInfo.peer_addr, peerMac, 6);
  peerInfo.channel = WIFI_CHANNEL; 
  peerInfo.encrypt = false;
  
  // Register peer
  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    #if DEBUG_LOCAL
    Serial.println("[ESP-NOW] Fatal: Failed to add External ESP32 peer.");
    #endif
    while (true) { delay(1000); } // Halt execution
  }

  // Initialize timers
  lastK210RxTime = millis();
}

void loop() {
  // Check for incoming serial data from the K210 on default Serial port (UART0)
  readFromK210();

  // Diagnostic LED non-blocking pulse timer
  // --- DISABLED PHYSICAL GPIO OUTPUTS AS PER REQUIREMENTS ---
  // if (ledActive && (millis() - pulseOnTime > 100)) {
  //   digitalWrite(2, LOW);
  //   ledActive = false;
  // }

  // Heartbeat check: If the K210 has been silent for more than 2 seconds,
  // send a "not detected" status message to the receiver as a keepalive packet.
  if (millis() - lastK210RxTime > HEARTBEAT_TIMEOUT_MS) {
    #if DEBUG_LOCAL
    Serial.println("[Heartbeat] K210 silent. Sending keepalive (NOT DETECTED) to peer.");
    #endif
    sendToExternal(false, -1, -1, -1);
    
    // Reset timer to avoid spamming packets continuously inside the loop
    lastK210RxTime = millis();
  }
}
