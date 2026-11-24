/*
  capture.ino - camera-to-LaTeX frame server for the EE 105 board
  (Arduino Nano 33 BLE Sense Lite + OV7675 camera).

  The board only takes pictures; recognition runs on the laptop.

  Serial protocol (USB; the baud rate setting does not matter on this board):
    'p'  ->  "OK <width> <height> 1\r\n"
    'c'  ->  "FRM1" | width u16 | height u16 | bpp u8 (=1) | count u32
             | count grayscale bytes, row by row | checksum u16
             (little-endian; checksum = sum of the pixel bytes mod 65536)
  If the camera does not start, the board prints "ERR camera" once a second.

  Board package: "Arduino Mbed OS Nano Boards", board "Arduino Nano 33 BLE".
  Library: Arduino_OV767X, version 0.0.2 or newer (it supports the OV7670 and
  the OV7675; the camera model is the fourth argument of Camera.begin).
*/

#include <Arduino_OV767X.h>

// ---- settings (bring-up step 3) ---------------------------------------------
#define CAMERA_MODEL  OV7675  // the EE 105 board carries an OV7675; write OV7670 for that module.
                          // Without it the library sets the sensor up as an OV7670.
#define USE_GRAYSCALE 1   // 1: ask the camera for GRAYSCALE; 0: ask for YUV422 and keep the Y bytes
#define Y_OFFSET      0   // which byte of each 2-byte pixel is brightness (0 or 1);
                          // flip it if the picture looks like noise or a negative
#define TEST_PATTERN  0   // 1: send the camera's colour-bar test pattern instead of the scene
#define CAMERA_FPS    1   // 1 is the most reliable; try 5 once the soak test passes
#define CHUNK         1024

static const int W = 320;                    // QVGA
static const int H = 240;
static const uint32_t COUNT = (uint32_t)W * H;
static uint8_t frame[COUNT * 2];             // room for 2 bytes per pixel (153,600 B)
static bool cameraOk = false;

static void writeU16(uint16_t v) {
  Serial.write((uint8_t)(v & 0xFF));
  Serial.write((uint8_t)(v >> 8));
}

static void writeU32(uint32_t v) {
  for (int i = 0; i < 4; i++) Serial.write((uint8_t)((v >> (8 * i)) & 0xFF));
}

// Did readFrame() write 2 bytes per pixel? The second half of the buffer is
// filled with a marker before reading; if the marker is gone, it did.
// (Camera.bytesPerPixel() is not trusted here: some library versions report 2
// in GRAYSCALE mode while already keeping only one byte per pixel.)
static bool wroteTwoBytes() {
  for (uint32_t k = 0; k < 64; k++) {
    if (frame[COUNT + k * (COUNT / 64) + 17] != 0xA5) return true;
  }
  return false;
}

static void sendFrame() {
  digitalWrite(LED_BUILTIN, HIGH);
  memset(frame + COUNT, 0xA5, COUNT);
  Camera.readFrame(frame);

  bool twoBytes = USE_GRAYSCALE ? wroteTwoBytes() : true;
  if (twoBytes) {
    // keep one brightness byte per pixel, in place (we read ahead of where we write)
    for (uint32_t i = 0; i < COUNT; i++) frame[i] = frame[2 * i + Y_OFFSET];
  }

  uint16_t sum = 0;
  for (uint32_t i = 0; i < COUNT; i++) sum += frame[i];

  Serial.write((const uint8_t *)"FRM1", 4);
  writeU16(W);
  writeU16(H);
  Serial.write((uint8_t)1);
  writeU32(COUNT);
  for (uint32_t off = 0; off < COUNT; off += CHUNK) {
    uint32_t n = (COUNT - off < CHUNK) ? (COUNT - off) : CHUNK;
    Serial.write(frame + off, n);
  }
  writeU16(sum);
  Serial.flush();
  digitalWrite(LED_BUILTIN, LOW);
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  while (!Serial) { }            // wait until the laptop opens the port

#if USE_GRAYSCALE
  cameraOk = Camera.begin(QVGA, GRAYSCALE, CAMERA_FPS, CAMERA_MODEL);
#else
  cameraOk = Camera.begin(QVGA, YUV422, CAMERA_FPS, CAMERA_MODEL);
#endif
#if TEST_PATTERN
  if (cameraOk) Camera.testPattern();
#endif
}

void loop() {
  if (!cameraOk) {
    Serial.println("ERR camera");
    delay(1000);
    return;
  }
  if (!Serial.available()) return;
  int c = Serial.read();
  if (c == 'p') {
    Serial.print("OK ");
    Serial.print(W);
    Serial.print(' ');
    Serial.print(H);
    Serial.println(" 1");
  } else if (c == 'c') {
    sendFrame();
  }
  // anything else (e.g. newlines from a serial terminal) is ignored
}
