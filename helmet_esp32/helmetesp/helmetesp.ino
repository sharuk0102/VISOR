/*
 * ==================================================================================
 * PROJECT: Sipeed Maixduino AprilTag Relay System
 * MODULE: Helmet ESP32 Sensor Sender (helmetesp)
 * BOARD: ESP32 (Generic, Arduino core)
 * LIBRARIES: <WiFi.h>, <esp_now.h>, <esp_wifi.h>
 * 
 * DESCRIPTION:
 *   This sketch runs on a secondary, helmet-mounted ESP32. It reads an analog
 *   alcohol sensor (MQ-3) and a Force Sensitive Resistor (FSR) pressure sensor.
 *   The readings are evaluated against thresholds, packaged into a structured
 *   payload, and sent to the central Receiver ESP32 via ESP-NOW.
 * 
 * WIRING SUMMARY:
 *   - MQ-3 Alcohol Sensor Analog Out   -> ESP32 GPIO 34 (ADC1_CH6)
 *   - FSR Pressure Sensor Analog Out   -> ESP32 GPIO 35 (ADC1_CH7)
 *   - VCC (MQ-3 / FSR)                 -> ESP32 5V / 3.3V (MQ-3 usually requires 5V for heater)
 *   - GND (MQ-3 / FSR)                 -> ESP32 GND
 * 
 * SENSOR CALIBRATION / WIRING DETAILS:
 *   - MQ-3: Has a built-in heater. It requires a pre-heating phase (usually 24-48 hours 
 *     for first use, and 1-2 minutes on boot) before readings stabilize.
 *     Connect the analog output (A0) to GPIO 34.
 *   - FSR: Connected in a voltage divider circuit. 
 *     Wiring: VCC (3.3V) -> FSR -> GPIO 35 -> 10k Ohm pull-down resistor -> GND.
 *     As pressure increases, the FSR resistance decreases, causing the voltage at 
 *     GPIO 35 to increase (0V when not worn, approaching VCC when heavily pressed).
 * 
 * TARGET MAC ADDRESS (For External ESP32 Receiver):
 *   - Station MAC: A0:B7:65:47:67:7C
 * ==================================================================================
 */

#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_arduino_version.h>

// --- PIN CONFIGURATION ────────────────────────────────────────────────────────────
const int ALCOHOL_PIN    = 34; // Analog input pin for MQ-3 alcohol sensor
const int FSR_PIN         = 35; // Analog input pin for FSR pressure sensor
const int HALL_PIN        = 32; // Analog input pin for Hall Effect sensor (strap lock)

// --- LED CONFIGURATION ────────────────────────────────────────────────────────────
const int RED_LED_PIN    = 25; // GPIO pin for RED LED
const int GREEN_LED_PIN  = 26; // GPIO pin for GREEN LED
const int BLUE_LED_PIN   = 27; // GPIO pin for BLUE LED (signal transfer status)

// --- CONSTANTS & CONFIGURATION ────────────────────────────────────────────────────
const unsigned long SERIAL_BAUD    = 115200; // Baud rate for Serial Monitor
const int WIFI_CHANNEL             = 1;      // WiFi Channel (MUST match external ESP32)
const unsigned long SEND_INTERVAL  = 500;    // Sensor transmission interval (500 ms)

// --- SENSOR THRESHOLDS ────────────────────────────────────────────────────────────
// Default thresholds (ESP32 ADC is 12-bit, range 0 to 4095). 
// Adjust these based on physical calibration.
const int ALCOHOL_THRESHOLD = 1900; // MQ-3 threshold: alert if MQ-3 > threshold
// FSR_TOUCH_THRESHOLD: Even a light/slight finger or head contact triggers PASS.
// 12-bit ADC range is 0-4095. A resting unloaded FSR reads near 0; a light touch
// typically reads 300-800. Start at 400 and tune down if too insensitive.
const int FSR_TOUCH_THRESHOLD  = 80;   // Very slight pressure triggers PASS (ultra-sensitive)

// FSR_REMOVE_THRESHOLD: Hard-floor near-zero reading that means helmet is clearly
// fully off. Must be well below TOUCH_THRESHOLD to avoid false FAILs from vibration.
const int FSR_REMOVE_THRESHOLD = 25;   // Only near-zero (heavy full removal) triggers FAIL

// FSR_HOLD_MS: Minimum window (ms) PASS state is held after a touch is detected.
// During this window minor pressure dips are ignored; only a sustained heavy
// removal (below FSR_REMOVE_THRESHOLD for FSR_REMOVAL_DEBOUNCE_MS) causes early FAIL.
const unsigned long FSR_HOLD_MS = 6000;  // 6-second hold window before re-evaluation

// FSR_REMOVAL_DEBOUNCE_MS: How long the reading must stay below FSR_REMOVE_THRESHOLD
// continuously before we declare FAIL. Prevents a single noisy ADC spike causing FAIL.
const unsigned long FSR_REMOVAL_DEBOUNCE_MS = 500; // 500 ms sustained heavy removal before FAIL (ignore fluctuations)
const int HALL_THRESHOLD     = 150;  // Hall effect magnet detection deviation threshold

// Magnetic baseline calibrated at startup
int hallBaseline = 2048;

// Destination Peer MAC Address (External ESP32 Station MAC)
uint8_t peerMac[] = {0x28, 0x05, 0xA5, 0xE2, 0xC6, 0x64}; // Receiver ESP32 MAC: 28:05:A5:E2:C6:64

// --- DATA STRUCTURES ──────────────────────────────────────────────────────────────
typedef struct {
  int alcoholValue;    // Raw ADC reading from MQ-3 (0 - 4095)
  int fsrValue;        // Raw ADC reading from FSR (0 - 4095)
  bool alcoholAlert;   // True if alcohol presence is detected (alcoholValue > threshold)
  bool fsrAlert;       // True if helmet is worn (fsrValue > threshold)
  uint32_t timestamp;  // Millisecond timestamp of transmission
} helmet_message;      // Size: 16 bytes

// --- GLOBAL VARIABLES ─────────────────────────────────────────────────────────────
unsigned long lastSendTime = 0;
volatile bool shouldPrintSentStatus = false; // State flag to avoid forwarding print spam

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
  // Turn Blue LED ON if transfer succeeded, OFF otherwise
  if (status == ESP_NOW_SEND_SUCCESS) {
    digitalWrite(BLUE_LED_PIN, HIGH);
  } else {
    digitalWrite(BLUE_LED_PIN, LOW);
  }

  if (shouldPrintSentStatus) {
    if (status == ESP_NOW_SEND_SUCCESS) {
      Serial.println("[HELMET ESP] Sent successfully");
    } else {
      Serial.println("[HELMET ESP] Send FAILED");
    }
    shouldPrintSentStatus = false;
  }
}

// --- MAIN FUNCTIONS ──────────────────────────────────────────────────────────────

void setup() {
  // Initialize Serial Monitor
  Serial.begin(SERIAL_BAUD);
  delay(1000);
  Serial.println("=================================================");
  Serial.println("Helmet ESP32 Sender Initializing...");
  Serial.println("=================================================");

  // Set sensor pins as analog inputs
  pinMode(ALCOHOL_PIN, INPUT);
  pinMode(FSR_PIN, INPUT);
  pinMode(HALL_PIN, INPUT);

  // Set LED pins as outputs
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(BLUE_LED_PIN, OUTPUT);

  // Initialize LEDs to low/off state
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(GREEN_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, LOW);

  // Calibrate Hall Effect sensor baseline (magnetic field reference)
  Serial.println("[HALL] Calibrating magnetic baseline...");
  long hallSum = 0;
  for (int i = 0; i < 20; i++) {
    hallSum += analogRead(HALL_PIN);
    delay(10);
  }
  hallBaseline = hallSum / 20;
  Serial.printf("[HALL] Baseline calibrated to: %d\n", hallBaseline);

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
  Serial.printf("[WiFi] Operating on Channel %d.\n", WIFI_CHANNEL);

  // Print current MAC Address for local reference
  Serial.print("[WiFi] Helmet ESP32 MAC Address: ");
  Serial.println(WiFi.macAddress());

  // Initialize ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("[ESP-NOW] Fatal: Initialization failed.");
    while (true) { delay(1000); } // Halt execution
  }
  Serial.println("[ESP-NOW] Initialized successfully.");

  // Register ESP-NOW Send Callback
  esp_now_register_send_cb(OnDataSent);

  // Configure Peer (External ESP32 Receiver)
  esp_now_peer_info_t peerInfo;
  memset(&peerInfo, 0, sizeof(peerInfo));
  memcpy(peerInfo.peer_addr, peerMac, 6);
  peerInfo.channel = WIFI_CHANNEL; 
  peerInfo.encrypt = false;
  
  // Register peer
  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("[ESP-NOW] Fatal: Failed to add External ESP32 peer.");
    while (true) { delay(1000); } // Halt execution
  }
  Serial.println("[ESP-NOW] External ESP32 peer added.");
  Serial.println("[HELMET ESP] Alcohol sensor + FSR initialized. Ready.");
  Serial.println("=================================================");
}

void loop() {
  // Read sensors and transmit at specified interval
  if (millis() - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = millis();

    // 1. Read analog sensors (with a 120-second warmup delay for the MQ-3)
    int rawAlcohol = 0;
    static bool warmupCompletedLogged = false;
    bool warmupCompletedNow = false;

    if (millis() > 120000) {
      rawAlcohol = analogRead(ALCOHOL_PIN);
      if (!warmupCompletedLogged) {
        warmupCompletedLogged = true;
        warmupCompletedNow = true;
        Serial.println("[HELMET ESP] MQ-3 Warmup Completed. Active reading started.");
      }
    } else {
      // Periodically print warmup status (every 10 seconds / 20 loops) to show life
      static int warmupPrints = 0;
      if (millis() / 10000 > warmupPrints) {
        warmupPrints = millis() / 10000;
        Serial.printf("[HELMET ESP] MQ-3 warming up... (%d seconds remaining)\n", 120 - (warmupPrints * 10));
      }
    }
    int rawFsr = analogRead(FSR_PIN);
    int rawHall = analogRead(HALL_PIN);

    // 2. Evaluate alert states with hysteresis to prevent threshold noise flickering
    static bool alcoholAlert = false;

    // ── FSR State Machine variables ───────────────────────────────────────────
    static bool fsrActive = false;              // Current PASS/FAIL state
    static unsigned long fsrDetectTime = 0;     // Timestamp when hold window started
    static bool holdingPass = false;            // True while inside 10-second hold window
    static unsigned long removalDetectTime = 0; // Timestamp when heavy-removal was first seen
    static bool removalPending = false;         // True if a sustained removal is being debounced

    // ── Alcohol hysteresis (unchanged) ───────────────────────────────────────
    if (!alcoholAlert) {
      if (rawAlcohol > ALCOHOL_THRESHOLD) {
        alcoholAlert = true;
      }
    } else {
      if (rawAlcohol < (ALCOHOL_THRESHOLD - 100)) {
        alcoholAlert = false;
      }
    }

    // ── FSR State Machine ─────────────────────────────────────────────────────
    //
    //  State IDLE  (holdingPass == false, fsrActive == false)
    //    Any reading >= FSR_TOUCH_THRESHOLD → transition to PASS immediately.
    //
    //  State PASS  (holdingPass == true, fsrActive == true)
    //    Holds for FSR_HOLD_MS.  During hold:
    //      • Minor dips (above FSR_REMOVE_THRESHOLD) → ignored (noise / riding vibration)
    //      • Heavy removal (below FSR_REMOVE_THRESHOLD) for FSR_REMOVAL_DEBOUNCE_MS
    //        continuously → early FAIL even before hold expires
    //    On hold expiry:
    //      • If FSR still >= FSR_TOUCH_THRESHOLD → restart hold (helmet still on)
    //      • Otherwise → FAIL
    //
    // ─────────────────────────────────────────────────────────────────────────

    if (holdingPass) {
      // We are inside the 10-second PASS hold window.
      fsrActive = true; // Default: keep PASS during this window

      // ── Early-FAIL check: heavy removal debounce ──────────────────────────
      // A hard-zero reading (well below any realistic wearing pressure) that
      // persists for FSR_REMOVAL_DEBOUNCE_MS means the helmet was actually removed.
      if (rawFsr < FSR_REMOVE_THRESHOLD) {
        // Reading is at hard-floor — start or continue the removal debounce timer
        if (!removalPending) {
          removalPending = true;          // First sub-threshold sample seen
          removalDetectTime = millis();   // Note when it started
        } else if (millis() - removalDetectTime >= FSR_REMOVAL_DEBOUNCE_MS) {
          // Sustained below-floor reading for debounce duration → genuine removal
          fsrActive    = false;
          holdingPass  = false;
          removalPending = false;
          Serial.println("[FSR] Heavy removal detected mid-hold → FAIL");
        }
        // else: still within debounce window, keep PASS for now
      } else {
        // Reading is above hard-floor: cancel any pending removal debounce
        // (noise spike or momentary lift — not a real removal)
        if (removalPending) {
          removalPending = false;
          Serial.println("[FSR] Removal debounce cancelled (pressure recovered)");
        }
      }

      // ── Hold expiry check ─────────────────────────────────────────────────
      if (holdingPass && (millis() - fsrDetectTime >= FSR_HOLD_MS)) {
        // 10-second window has elapsed — re-evaluate fresh pressure
        holdingPass    = false;
        removalPending = false;
        if (rawFsr >= FSR_TOUCH_THRESHOLD) {
          // Helmet still registering contact → restart hold, stay PASS
          fsrActive      = true;
          fsrDetectTime  = millis();
          holdingPass    = true;
          Serial.println("[FSR] Hold expired, pressure sustained → hold restarted");
        } else {
          // No meaningful pressure at re-evaluation → FAIL
          fsrActive = false;
          Serial.println("[FSR] Hold expired, no pressure → FAIL");
        }
      }

    } else {
      // ── IDLE / FAIL state: watch for any light touch to enter PASS ────────
      removalPending = false; // Clear any stale debounce state
      if (rawFsr >= FSR_TOUCH_THRESHOLD) {
        // Even a slight touch is enough to declare helmet worn
        fsrActive     = true;
        fsrDetectTime = millis(); // Start the 10-second hold window now
        holdingPass   = true;
        Serial.printf("[FSR] Touch detected (raw=%d) → PASS, hold started\n", rawFsr);
      } else {
        fsrActive = false; // No pressure at all → stay FAIL
      }
    }

    // Hall Effect strap lock detection: (BYPASSED — always PASS)
    bool strapLocked = true;

    // Combined Helmet Worn Condition (FSR + Strap Lock)
    bool fsrAlert = fsrActive && strapLocked;

    // Update Red/Green LED indicator state:
    // Red LED glows if alcohol is detected OR helmet is not worn (FSR not detected).
    // Green LED glows if helmet is worn (FSR detected) AND alcohol is not detected.
    if (alcoholAlert || !fsrAlert) {
      digitalWrite(RED_LED_PIN, HIGH);
      digitalWrite(GREEN_LED_PIN, LOW);
    } else {
      digitalWrite(RED_LED_PIN, LOW);
      digitalWrite(GREEN_LED_PIN, HIGH);
    }

    // 3. Package details into the payload structure
    helmet_message msg;
    msg.alcoholValue = rawAlcohol;
    msg.fsrValue = rawFsr;
    msg.alcoholAlert = alcoholAlert;
    msg.fsrAlert = fsrAlert;
    msg.timestamp = millis();

    // Print status messages based on alerts (ONLY on state change)
    static bool lastAlcAlert = false;
    static bool lastFsrAlert = false;
    static bool firstSensorUpdate = true;

    bool stateChanged = firstSensorUpdate || (alcoholAlert != lastAlcAlert) || (fsrAlert != lastFsrAlert) || warmupCompletedNow;

    if (stateChanged) {
      Serial.printf("[HELMET ESP] Alcohol: %d (Level: %d)\n", alcoholAlert ? 1 : 0, rawAlcohol);

      Serial.print("[HELMET ESP] FSR Pressure: ");
      if (rawFsr < FSR_REMOVE_THRESHOLD) {
        Serial.printf("Low Pressure (Value: %d)\n", rawFsr);
      } else if (rawFsr <= FSR_TOUCH_THRESHOLD) {
        Serial.printf("Medium Pressure (Value: %d)\n", rawFsr);
      } else {
        Serial.printf("High Pressure (Value: %d)\n", rawFsr);
      }

      Serial.printf("[HELMET ESP] FSR Active: %d, Strap Locked: %d\n", fsrActive ? 1 : 0, strapLocked ? 1 : 0);
      Serial.printf("[HELMET ESP] Combined Status (FSR + Strap): %d\n", fsrAlert ? 1 : 0);

      if (fsrAlert) {
        Serial.println("[HELMET ESP] Helmet Worn & Strap Locked (PASS)");
      } else {
        Serial.println("[HELMET ESP] Helmet Not Worn or Strap Unlocked (FAIL)");
      }
    }

    // 4. Send packet via ESP-NOW
    if (stateChanged) {
      Serial.println("[HELMET ESP] Sending data to External ESP32...");
      shouldPrintSentStatus = true;
    } else {
      shouldPrintSentStatus = false;
    }

    esp_err_t result = esp_now_send(peerMac, (uint8_t *)&msg, sizeof(msg));
    if (result != ESP_OK && stateChanged) {
      Serial.println("[HELMET ESP] Sending data to External ESP32... Send FAILED");
    }

    lastAlcAlert = alcoholAlert;
    lastFsrAlert = fsrAlert;
    firstSensorUpdate = false;
  }
}
