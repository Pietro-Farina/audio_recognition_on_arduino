/*
    This example reads audio data from the on-board PDM microphones, and prints
    out the samples to the Serial console. The Serial Plotter built into the
    Arduino IDE can be used to plot the audio data (Tools -> Serial Plotter)
    Circuit:
    - Arduino Nicla Vision, or
    - Arduino Nano 33 BLE board, or
    - Arduino Nano RP2040 Connect, or
    - Arduino Portenta H7 board plus Portenta Vision Shield
    This example code is in the public domain.
  */

#include <PDM.h>

// default number of output channels
static const char channels = 1;

// default PCM output frequency
static const int frequency = 16000;
static const int recordSamples = 16000;

// Buffer to read samples into, each sample is 16-bits
short recordBuffer[recordSamples];
short sampleBuffer[512];

// Number of audio samples read
volatile int samplesRead;
volatile bool recordingDone = false;
volatile bool recording = false;
volatile int recordIndex = 0;

// Blinking
bool state = false;
int timeStart = 0;

void setup() {
    Serial.begin(115200);
    pinMode(LEDB, OUTPUT);

    while (!Serial);

    // Configure the data receive callback
    PDM.onReceive(onPDMdata);

    // Optionally set the gain
    // Defaults to 20 on the BLE Sense and 24 on the Portenta Vision Shield
    // PDM.setGain(30);

    // Initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate for the Arduino Nano 33 BLE Sense
    // - a 32 kHz or 64 kHz sample rate for the Arduino Portenta Vision Shield
    if (!PDM.begin(channels, frequency)) {
        Serial.println("Failed to start PDM!");
        while (1);
    }
}

void loop() {
    // ---- Trigger ----
    if (Serial.available()) {
        char c = Serial.read();
        while (Serial.available()) Serial.read();  // flush

        if (c == 'r') {
            recordIndex = 0;
            Serial.println("Recording...");
            recording = true;
        }
    }

    // Wait for samples to be read
    if (recording && samplesRead) {
        for (int i = 0; i < samplesRead; i++) {
            if (recordIndex < recordSamples) {
                recordBuffer[recordIndex++] = sampleBuffer[i];
            } else {
                recording = false;
                Serial.println("DONE");
                delay(1000);
                sendRecording();
                break;
            }
        }
    }
    samplesRead = 0;
}

void sendRecording() {
    for (int i = 0; i < recordSamples; i++) {
        Serial.println(recordBuffer[i]);
    }
}

/**
  Callback function to process the data from the PDM microphone.
  NOTE: This callback is executed as part of an ISR.
  Therefore using `Serial` to print messages inside this function isn't
supported.
* */
void onPDMdata() {
    // Query the number of available bytes
    int bytesAvailable = PDM.available();

    // Read into the sample buffer
    PDM.read(sampleBuffer, bytesAvailable);

    // 16-bit, 2 bytes per sample
    samplesRead = bytesAvailable / 2;
}