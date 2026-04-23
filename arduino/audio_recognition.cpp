/*
    This example reads audio data from the on-board PDM microphones, and converts it
    into MFCC features to run inference on a gesture recognition model.
  */

#include <PDM.h>

// For running the neural network
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// For the feature extraction pipeline
#include <arm_const_structs.h>
#include <arm_math.h>
// This are the precomputed parameters for the MFCC feature extraction, such as
// the mel filterbank, DCT matrix and Hamming window float window[], float
// mel_fb[], dct[]
#include "precomputed_params.h"

// Model
#include "model.h"

// Constant for MFCC pipeline
#define FRAME_LEN 256
#define HOP_LEN 128  // 50% overlap
#define NUM_MEL 26
#define NUM_MFCC 13
#define TOTAL_FRAMES 125

// PCM constants
static const char channels = 1;
static const int frequency = 16000;

// Memory Management: pdm buffer, frame buffer for MFCC, feature matrix for
// model input
short sampleBuffer[HOP_LEN];
float32_t frame[FRAME_LEN] = {
    0.0f};  // We initialize to avoid problems with the first memory slide
float32_t featureMatrix[TOTAL_FRAMES][NUM_MFCC];  // store input of the network

// Variables for MFCC pipeline
float32_t preemph[FRAME_LEN];
float32_t windowed[FRAME_LEN];
float32_t fft_out[FRAME_LEN];
float32_t power[129];
float32_t mel_energy[NUM_MEL];
float32_t mfcc[NUM_MFCC];
arm_rfft_fast_instance_f32 rfft;

// Number of audio samples read
volatile int samplesRead;
int frameCount = 0;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to be adjusted
// based on the model you are using
constexpr int tensorArenaSize = 64 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* GESTURES[] = {"clap", "silence", "snap", "tap"};

#define NUM_GESTURES 4

#define AUDIO_BUFFER_SIZE 1024  // must be > FRAME_LEN

volatile int16_t audioBuffer[AUDIO_BUFFER_SIZE];
volatile int writeIndex = 0;
volatile int readIndex = 0;

void mfcc_process() {
    // 1. Pre-emphasis
    preemph[0] = frame[0];
    for (int i = 1; i < FRAME_LEN; i++) {
        preemph[i] = frame[i] - 0.97f * frame[i - 1];
    }

    // 2. Window
    arm_mult_f32(preemph, window, windowed, FRAME_LEN);

    // 3. FFT
    arm_rfft_fast_f32(&rfft, windowed, fft_out, 0);

    // 4. Power spectrum
    power[0] = fft_out[0] * fft_out[0];    // DC
    power[128] = fft_out[1] * fft_out[1];  // Nyquist

    arm_cmplx_mag_squared_f32(&fft_out[2], &power[1], 127);

    // 5. Mel filterbank
    for (int m = 0; m < NUM_MEL; m++) {
        arm_dot_prod_f32(&mel_fb[m * 129], power, 129, &mel_energy[m]);

        mel_energy[m] = logf(mel_energy[m] + 1e-10f);

        if (isnan(mel_energy[m]) || isinf(mel_energy[m])) {
            mel_energy[m] = 0.0f;
        }
    }

    // 6. DCT
    for (int k = 0; k < NUM_MFCC; k++) {
        arm_dot_prod_f32(&dct[k * NUM_MEL], mel_energy, NUM_MEL, &mfcc[k]);
    }
}

void setup() {
    Serial.begin(115200);

    while (!Serial);

    // get the TFL representation of the model byte array
    tflModel = tflite::GetModel(model);
    if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        while (1);
    }

    // Create an interpreter to run the model
    tflInterpreter =
        new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena,
                                     tensorArenaSize, &tflErrorReporter);

    // Allocate memory for the model's input and output tensors
    TfLiteStatus allocate_status = tflInterpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.print("Error: AllocateTensors() failed! Status: ");
        Serial.println(allocate_status);
        // If this prints, you need to increase tensorArenaSize
        while (1);
    }
    Serial.println("Tensors allocated successfully!");

    // Get pointers for the model's input and output tensors
    tflInputTensor = tflInterpreter->input(0);
    tflOutputTensor = tflInterpreter->output(0);

    // Initialize fft
    arm_rfft_fast_init_f32(&rfft, FRAME_LEN);

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

    Serial.println("PDM microphone started.");
}

void loop() {
    int16_t hopBuffer[HOP_LEN];

    // wait until enough samples exist
    if (!readSamples(hopBuffer, HOP_LEN)) return;

    if (true) {
        samplesRead = 0;  // Reset immediately to avoid double-processing

        //  1: Shift the frame buffer to maintain overlap, move old 128 values
        //  to the start
        memmove(frame, &frame[HOP_LEN], HOP_LEN * sizeof(float32_t));

        // 2: Fill the rest with the new data from the PDM
        for (int i = 0; i < HOP_LEN; i++) {
            frame[HOP_LEN + i] = (float32_t)hopBuffer[i];
        }

        // 3: Perform MFCC pipeline on the frame
        mfcc_process();

        // 4: Store the MFCC result in the feature matrix
        if (frameCount < TOTAL_FRAMES) {
            // copy mfcc to feature matrix
            for (int i = 0; i < NUM_MFCC; i++) {
                featureMatrix[frameCount][i] = mfcc[i];
            }
            frameCount++;
        }

        // 5: if matrix full, trigger model inference
        if (frameCount == TOTAL_FRAMES) {
            // Copy feature matrix to model input
            memcpy(tflInputTensor->data.f, featureMatrix,
                   sizeof(featureMatrix));

            // Run inference
            if (tflInterpreter->Invoke() == kTfLiteOk) {
                for (int i = 0; i < NUM_GESTURES; i++) {
                    Serial.print(GESTURES[i]);
                    Serial.print(": ");
                    Serial.println(tflOutputTensor->data.f[i], 3);
                }
                Serial.println("---");
            } else {
                Serial.println("Inference failed!");
                return;
            }

            // After inference, we shift the last 63 frames to the start of the
            // feature matrix to create a sliding window effect for continuous
            // inference memmove(&featureMatrix[0], &featureMatrix[62], 63 *
            // NUM_MFCC * sizeof(float)); frameCount = 63;
            frameCount = 0;
        }
    }
}

bool readSamples(int16_t* dest, int n) {
    if (availableSamples() < n) return false;

    for (int i = 0; i < n; i++) {
        dest[i] = audioBuffer[readIndex];
        readIndex = (readIndex + 1) % AUDIO_BUFFER_SIZE;
    }
    return true;
}

int availableSamples() {
    if (writeIndex >= readIndex)
        return writeIndex - readIndex;
    else
        return AUDIO_BUFFER_SIZE - (readIndex - writeIndex);
}

/**
  Callback function to process the data from the PDM microphone.
  NOTE: This callback is executed as part of an ISR.
  Therefore using `Serial` to print messages inside this function isn't
supported.
* */
void onPDMdata() {
    int bytesAvailable = PDM.available();
    int samples = bytesAvailable / 2;

    int16_t temp[256];  // temp chunk
    PDM.read(temp, bytesAvailable);

    for (int i = 0; i < samples; i++) {
        int nextWrite = (writeIndex + 1) % AUDIO_BUFFER_SIZE;

        // prevent overwrite (drop oldest if full)
        if (nextWrite != readIndex) {
            audioBuffer[writeIndex] = temp[i];
            writeIndex = nextWrite;
        }
    }
}