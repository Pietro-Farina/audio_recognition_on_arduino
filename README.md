# Audio Recognition on Arduino

This project focuses on implementing **Mel-Frequency Cepstral Coefficients (MFCC)** on Arduino for embedded audio processing using the CMSIS-DSP framework.

The goal is to acquire audio from a microphone, process it directly on the Arduino board, extract MFCC features through a CMSIS-based pipeline, and use these features for simple sound or keyword recognition.

---

## Overview

The project is divided into three main stages:

* **Data acquisition**
  Audio is recorded from a microphone using an Arduino device to build a dataset for training.

* **Data processing and model training**
  A Jupyter notebook is used to:

  * Organize and preprocess the dataset
  * Extract MFCC features
  * Train and evaluate a classification model
  * Convert the trained model into a TensorFlow Lite Micro (`.tflite`) format

* **On-device inference**
  An Arduino sketch runs the trained model and performs real-time inference on incoming audio.

---

## Project Structure

```
├── README.md                         # Project documentation
├── On_device_Keyword_Spotting.ipynb  # Data processing and model training
├── dataset/                          # Audio samples + recording sketch
├── arduino/                          # Arduino inference code
```

---

## Model Training and Evaluation

We follow a standard MFCC extraction pipeline to transform raw PDM audio data into features suitable for a neural network:

1. **Audio sampling**
   Audio is captured on-device using PDM at 16 kHz.

2. **Frame segmentation**
   Each 1-second sample is divided into overlapping frames of 256 samples.

3. **Pre-emphasis**
   A filter boosts high frequencies to improve FFT performance.

4. **Hamming window**
   Applied to each frame to reduce spectral artifacts caused by sharp edges.

5. **RFFT computation**
   A Real Fast Fourier Transform converts each frame into 129 frequency bins.

6. **Mel filter bank**
   26 filters group frequencies according to human auditory perception.

7. **Log energy computation**
   Logarithmic scaling compresses dynamic range across frequency bands.

8. **DCT (Discrete Cosine Transform)**
   DCT-II decorrelates features, producing 26 coefficients.

9. **MFCC selection**
   The first 13 coefficients (C1–C13) are retained as features.

### Output Representation

From a 1-second audio sample (16,000 values), we obtain:

* **125 frames × 13 coefficients → (125 × 13) feature matrix**

This matrix is used as input to the neural network.

### Optimization

To improve efficiency during inference, several components are precomputed and stored:

* Hamming window
* Mel filter banks
* DCT coefficients

---

## Model Architecture

The trained model is a compact Convolutional Neural Network (CNN) designed for embedded deployment:

* 3 convolutional layers
* Pooling layers to reduce dimensionality
* Dropout layers to prevent overfitting

Despite its small size, the model generalizes well and performs reliably on unseen data.

---

## Arduino Deployment

The Arduino implementation mirrors the training pipeline:

* PDM audio is continuously collected into a circular buffer
* Once enough samples are available, the MFCC pipeline is executed
* Extracted features are accumulated until a full input tensor is ready
* The tensor is passed to the neural network for inference
* Results (class scores) are printed via Serial

### Notes

* Continuous inference using overlapping feature windows was tested but discarded due to excessive computation and reduced result clarity.
* Deployment requires including:

  * `model.h`
  * `precomputed_params.h`

---

## Results and Considerations

The model demonstrates strong performance despite the limited dataset. It successfully distinguishes between gestures even when recordings vary slightly in execution.

This highlights:

* The robustness of MFCC features
* The effectiveness of lightweight CNNs in embedded applications

---

## Future Improvements

* Expand dataset size and diversity
* Add real-time feedback or actuation based on predictions

---