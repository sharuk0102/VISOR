/*
 * ==================================================================================
 * PROJECT: Sipeed Maixduino AprilTag Relay System
 * MODULE: Receiver ESP32 Sketch (receiveresp)
 * BOARD: ESP32 (Generic, Arduino core)
 * LIBRARIES: <WiFi.h>, <esp_now.h>, <esp_wifi.h>
 * 
 * WIRING SUMMARY:
 *   - Onboard LED -> GPIO 2 (Status indicator)
 *   - USB Serial -> Debug output to PC (115200 baud)
 * 
 * TARGET MAC ADDRESS (For this External ESP32):
 *   - Station MAC: 28:05:A5:E2:C6:64
 * ==================================================================================
 */

#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_arduino_version.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// --- CONSTANTS & CONFIGURATION ────────────────────────────────────────────────────
const unsigned long DEBUG_BAUD   = 115200; // Baud rate for USB Debug Serial
const int LED_PIN                = 2;      // Onboard LED GPIO Pin (usually 2 on generic ESP32)
const int WIFI_CHANNEL           = 1;      // WiFi Channel (MUST match internal ESP32)
const unsigned long LINK_TIMEOUT_MS = 5000; // Time (ms) after which link is considered lost

// --- MOTOR DRIVER PINS (L298N) ────────────────────────────────────────────────────
// Connect ESP32 GPIO 12 to L298N IN1 direction pin
// Connect ESP32 GPIO 13 to L298N IN2 direction pin
// Connect ESP32 GPIO 14 to L298N ENA speed pin
// Connect ESP32 GND to L298N GND (common ground)
const int MOTOR_IN1 = 12;
const int MOTOR_IN2 = 13;
const int MOTOR_ENA = 14;

// Motor speed constants (PWM duty cycles 0-255)
const int SPEED_LIMITED = 100;  // PWM value for restricted/safe speed (slow — conditions not met)
const int SPEED_MAXIMUM = 230;  // PWM value for unlocked/full speed (fast — all conditions PASS)

// --- DATA STRUCTURES ──────────────────────────────────────────────────────────────
typedef struct {
  bool detected;       // True if tag is detected, false otherwise
  int tagID;           // The ID of the detected tag (-1 if not detected)
  int centerX;         // X center coordinate (-1 if not detected/applicable)
  int centerY;         // Y center coordinate (-1 if not detected/applicable)
  uint32_t timestamp;  // Millisecond timestamp of transmission
} tag_message;

typedef struct {
  int alcoholValue;    // Raw ADC reading from MQ-3 (0 - 4095)
  int fsrValue;        // Raw ADC reading from FSR (0 - 4095)
  bool alcoholAlert;   // True if alcohol presence is detected (alcoholValue > threshold)
  bool fsrAlert;       // True if helmet is worn (fsrValue > threshold)
  uint32_t timestamp;  // Millisecond timestamp of transmission
} helmet_message;      // Size: 16 bytes

// --- GLOBAL VARIABLES ─────────────────────────────────────────────────────────────
volatile tag_message receivedData;          // Buffer for incoming AprilTag packet
volatile uint32_t lastPacketTime = 0;      // Timestamp of the last received AprilTag packet
volatile bool newPacketReceived = false;   // Flag to alert loop of a new AprilTag packet
bool linkLost = false;                     // Status of the AprilTag wireless connection

volatile helmet_message receivedHelmetData;      // Buffer for incoming helmet sensor packet
volatile uint32_t lastHelmetPacketTime = 0;      // Timestamp of the last received helmet packet
volatile bool newHelmetPacketReceived = false;   // Flag to alert loop of a new helmet packet
bool helmetLinkLost = false;                     // Status of the helmet sensor wireless connection

// --- ESP-NOW CORE VERSION COMPATIBILITY ───────────────────────────────────────────
#if defined(ESP_ARDUINO_VERSION) && (ESP_ARDUINO_VERSION >= ESP_ARDUINO_VERSION_VAL(3, 0, 0))
  #define ESP_NOW_CORE_V3
#endif

// --- BLE CONFIGURATION & STATE ───────────────────────────────────────────────────
#define SERVICE_UUID         "12345678-1234-1234-1234-1234567890AB"
#define HELMET_CHAR_UUID     "BEB5483E-36E1-4688-B7F5-EA07361B26A8"
#define STRAP_CHAR_UUID      "CED11C71-7030-4C10-90D1-64582E5507E8"
#define SPEED_CHAR_UUID      "DFB5483E-36E1-4688-B7F5-EA07361B26A8"

BLECharacteristic *pHelmetChar = nullptr;
BLECharacteristic *pStrapChar = nullptr;
BLECharacteristic *pSpeedChar = nullptr;
volatile bool deviceConnected = false;

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("[BLE] App connected 📱");
    }

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      Serial.println("[BLE] App disconnected. Restarting advertising...");
      pServer->startAdvertising(); // restart advertising
    }
};

// ESP-NOW Receive Callback
#if defined(ESP_NOW_CORE_V3)
void OnDataRecv(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len) {
  const uint8_t *mac_addr = info->src_addr;
#else
void OnDataRecv(const uint8_t *mac_addr, const uint8_t *incomingData, int len) {
#endif
  if (len == sizeof(tag_message)) {
    // Copy the payload into the global struct
    memcpy((void*)&receivedData, incomingData, sizeof(receivedData));
    lastPacketTime = millis();
    newPacketReceived = true;
    linkLost = false;
  } 
  else if (len == sizeof(helmet_message)) {
    // Copy the payload into the helmet global struct
    memcpy((void*)&receivedHelmetData, incomingData, sizeof(receivedHelmetData));
    lastHelmetPacketTime = millis();
    newHelmetPacketReceived = true;
    helmetLinkLost = false;
  } 
  else {
    Serial.println("[EXTERNAL ESP] Unknown packet received, ignoring.");
  }
}

// --- HELPER FUNCTIONS ─────────────────────────────────────────────────────────────

// Updates the state of the status LED
void updateStatusLED(bool detected) {
  if (detected) {
    // Tag detected: Solid ON
    digitalWrite(LED_PIN, HIGH);
  } else {
    // Tag not detected or Link Lost: OFF
    digitalWrite(LED_PIN, LOW);
  }
}

// --- MAIN FUNCTIONS ──────────────────────────────────────────────────────────────

void setup() {
  // Initialize USB Serial for PC debug prints
  Serial.begin(DEBUG_BAUD);
  delay(1000); // Allow Serial port to initialize
  Serial.println("=================================================");
  Serial.println("Receiver ESP32 (receiveresp) Initializing...");
  Serial.println("=================================================");

  // Initialize Pin
  // --- DISABLED PHYSICAL GPIO OUTPUTS AS PER REQUIREMENTS ---
  // pinMode(LED_PIN, OUTPUT);
  // digitalWrite(LED_PIN, LOW); // Start with LED off

  // Configure L298N Motor Driver Pins
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  pinMode(MOTOR_ENA, OUTPUT);
  digitalWrite(MOTOR_IN1, HIGH); // Motor configured for forward direction
  digitalWrite(MOTOR_IN2, LOW);
  analogWrite(MOTOR_ENA, SPEED_LIMITED); // Default to limited speed (20 km/h / 100 RPM)

  // Initialize WiFi in Station mode
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(); // Disconnect from any saved network profile

  // Start the WiFi interface (Required before setting the channel manually)
#if defined(ESP_NOW_CORE_V3)
  esp_wifi_start();
#endif

  // Force WiFi to operate on a fixed channel (MUST match internal ESP32)
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);
  Serial.printf("[WiFi] Operating on Channel %d.\n", WIFI_CHANNEL);

  // Print current MAC Address for confirmation
  Serial.print("[WiFi] Current STA MAC Address: ");
  Serial.println(WiFi.macAddress());

  // Initialize ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("[ESP-NOW] Fatal: Initialization failed.");
    while (true) { delay(1000); } // Halt execution
  }
  Serial.println("[ESP-NOW] Initialized successfully.");

  // Register ESP-NOW Receive Callback
  esp_now_register_recv_cb(OnDataRecv);

  // Initialize BLE
  Serial.println("[BLE] Initializing BLE Server...");
  BLEDevice::init("ESP32_HELMET_RECEIVER");
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Create Helmet Characteristic (Vision / AprilTag status)
  pHelmetChar = pService->createCharacteristic(
                  HELMET_CHAR_UUID,
                  BLECharacteristic::PROPERTY_READ |
                  BLECharacteristic::PROPERTY_NOTIFY
                );
  pHelmetChar->addDescriptor(new BLE2902());

  // Create Strap Characteristic (FSR / Head presence status)
  pStrapChar = pService->createCharacteristic(
                 STRAP_CHAR_UUID,
                 BLECharacteristic::PROPERTY_READ |
                 BLECharacteristic::PROPERTY_NOTIFY
               );
  pStrapChar->addDescriptor(new BLE2902());

  // Create Speed Characteristic (Motor Speed / Telemetry status)
  pSpeedChar = pService->createCharacteristic(
                 SPEED_CHAR_UUID,
                 BLECharacteristic::PROPERTY_READ |
                 BLECharacteristic::PROPERTY_NOTIFY
               );
  pSpeedChar->addDescriptor(new BLE2902());

  pService->start();

  // Configure BLE Advertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);  // helps with iOS connection issues
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();
  Serial.println("[BLE] Advertising started. Ready for App connection.");

  // Initialize Timers
  lastPacketTime = millis();
  lastHelmetPacketTime = millis();
  Serial.println("[EXTERNAL ESP] Ready. Listening for Internal ESP32 and Helmet ESP32...");
  Serial.println("=================================================");
}

void loop() {
  static bool lastState = false;
  static bool firstStateUpdate = true;

  // 1. Process newly arrived AprilTag packet
  if (newPacketReceived) {
    newPacketReceived = false;

    // Read volatile struct details in local variables safely
    bool tagDetected = receivedData.detected;
    int tagID = receivedData.tagID;

    // Print received status to Serial (ONLY on state change)
    static bool lastTagDetected = false;
    static int lastTagIDVal = -2;
    static bool firstTagPrint = true;

    if (firstTagPrint || (tagDetected != lastTagDetected) || (tagDetected && tagID != lastTagIDVal)) {
      if (tagDetected) {
        Serial.println("from sipeed: detected");
      } else {
        Serial.println("from sipeed: not detected");
      }
      lastTagDetected = tagDetected;
      lastTagIDVal = tagID;
      firstTagPrint = false;
    }

    // Update status LED
    // --- DISABLED PHYSICAL GPIO OUTPUTS AS PER REQUIREMENTS ---
    // updateStatusLED(tagDetected);
  }

  // 2. Process newly arrived helmet sensor packet
  if (newHelmetPacketReceived) {
    newHelmetPacketReceived = false;

    // Read volatile struct details in local variables safely
    int rawAlc = receivedHelmetData.alcoholValue;
    int rawFsr = receivedHelmetData.fsrValue;
    bool alcAlert = receivedHelmetData.alcoholAlert;
    bool fsrWorn = receivedHelmetData.fsrAlert;

    // Print clearly which source each incoming packet came from (ONLY on state change)
    static bool lastAlcAlertVal = false;
    static bool lastFsrWornVal = false;
    static bool firstHelmetPrint = true;

    if (firstHelmetPrint || (alcAlert != lastAlcAlertVal) || (fsrWorn != lastFsrWornVal)) {
      Serial.printf("from helmet esp\nfsr:%s  alcohol:%s\n",
                    fsrWorn ? "detected" : "not detected",
                    alcAlert ? "detected" : "absent");
      lastAlcAlertVal = alcAlert;
      lastFsrWornVal = fsrWorn;
      firstHelmetPrint = false;
    }
  }

  // Combined Alert Logic Evaluation
  static bool lastAlertTriggered = false;
  static String lastAlertReason = "";
  static bool firstAlertCheck = true;

  bool riderNotWearing = (!receivedData.detected || !receivedHelmetData.fsrAlert);
  bool alcoholDetected = receivedHelmetData.alcoholAlert;

  bool alertTriggered = false;
  String alertReason = "";

  if (!linkLost && !helmetLinkLost) {
    if (riderNotWearing && alcoholDetected) {
      alertTriggered = true;
      alertReason = "Rider not wearing helmet + Alcohol detected";
    } else if (riderNotWearing) {
      alertTriggered = true;
      alertReason = "Rider not wearing helmet";
    } else if (alcoholDetected) {
      alertTriggered = true;
      alertReason = "Alcohol detected";
    }
  }

  if (firstAlertCheck || (alertTriggered != lastAlertTriggered) || (alertTriggered && alertReason != lastAlertReason)) {
    if (alertTriggered) {
      Serial.printf("ALERT: %s\n", alertReason.c_str());
    } else {
      if (!linkLost && !helmetLinkLost) {
        Serial.println("speed lock open have a safe ride 🏍");
      }
    }
    lastAlertTriggered = alertTriggered;
    lastAlertReason = alertReason;
    firstAlertCheck = false;
  }

  // 3. Connection check for AprilTag sender (Internal ESP32)
  if (!linkLost && (millis() - lastPacketTime > LINK_TIMEOUT_MS)) {
    Serial.println("[EXTERNAL ESP] LINK LOST - No data from Internal ESP32");
    linkLost = true;
    // --- DISABLED PHYSICAL GPIO OUTPUTS AS PER REQUIREMENTS ---
    // updateStatusLED(false); // Force LED off on connection drop
    firstStateUpdate = true; // Reset state tracking so new status prints instantly upon link re-establishment
    firstAlertCheck = true;  // Reset alert logic tracking for re-connection print
  }

  // 4. Connection check for Helmet sensor sender (helmetesp)
  if (!helmetLinkLost && (millis() - lastHelmetPacketTime > LINK_TIMEOUT_MS)) {
    Serial.println("[EXTERNAL ESP] LINK LOST - No data from Helmet ESP32");
    helmetLinkLost = true;
    firstAlertCheck = true;  // Reset alert logic tracking for re-connection print
  }

  // 5. Motor Output Control (L298N + BO Motor)
  // System is Safe ONLY when both links are active, helmet is worn, and no alcohol is detected.
  static int currentMotorPWM = SPEED_LIMITED;
  if (!linkLost && !helmetLinkLost && !riderNotWearing && !alcoholDetected) {
    // Turn Motor ON at MAXIMUM SPEED (Safe)
    digitalWrite(MOTOR_IN1, HIGH);
    digitalWrite(MOTOR_IN2, LOW);
    currentMotorPWM = SPEED_MAXIMUM;
    analogWrite(MOTOR_ENA, SPEED_MAXIMUM);
  } else {
    // Turn Motor ON at LIMITED SPEED (Restricted / Unsafe)
    digitalWrite(MOTOR_IN1, HIGH);
    digitalWrite(MOTOR_IN2, LOW);
    currentMotorPWM = SPEED_LIMITED;
    analogWrite(MOTOR_ENA, SPEED_LIMITED);
  }

  // 6. Update BLE App notifications on state change
  static bool lastBleHelmet = false;
  static bool lastBleStrap = false;
  static uint8_t lastBleSpeed = 0;
  static bool firstBleSend = true;

  if (deviceConnected) {
    // Map status:
    // - Helmet Worn (vision detected) requires active link and AprilTag detection
    // - Head/Strap present (physical sensors) requires active link and FSR detection
    bool currentBleHelmet = (!linkLost && receivedData.detected);
    bool currentBleStrap = (!helmetLinkLost && receivedHelmetData.fsrAlert);

    // Compute speed (km/h) based on motor ENA PWM output:
    // Swapped to match the physical active-low motor driver configuration:
    // - SPEED_MAXIMUM (100 PWM, fast speed on active-low ENA) -> 46 km/h
    // - SPEED_LIMITED (230 PWM, slow speed on active-low ENA) -> 20 km/h
    uint8_t currentSpeedKmh = (currentMotorPWM == SPEED_MAXIMUM) ? 46 : 20;

    if (firstBleSend || (currentBleHelmet != lastBleHelmet) || (currentBleStrap != lastBleStrap) || (currentSpeedKmh != lastBleSpeed)) {
      uint8_t helmetVal = currentBleHelmet ? 1 : 0;
      uint8_t strapVal = currentBleStrap ? 1 : 0;

      pHelmetChar->setValue(&helmetVal, 1);
      pHelmetChar->notify();

      pStrapChar->setValue(&strapVal, 1);
      pStrapChar->notify();

      pSpeedChar->setValue(&currentSpeedKmh, 1);
      pSpeedChar->notify();

      lastBleHelmet = currentBleHelmet;
      lastBleStrap = currentBleStrap;
      lastBleSpeed = currentSpeedKmh;
      firstBleSend = false;
      Serial.printf("[BLE] Notified app: HelmetWorn=%d, HeadPresent=%d, SpeedKmh=%d\n", helmetVal, strapVal, currentSpeedKmh);
    }
  } else {
    firstBleSend = true; // Reset tracker when app disconnected so we notify initial values on reconnect
  }
}
