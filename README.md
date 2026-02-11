# Bird Sound Classification using Transfer Learning (EfficientNetB0)

## Project Overview  
This project builds an end-to-end machine learning pipeline to classify bird species from **audio spectrogram images**.  
Raw bird audio recordings were collected from **Xeno-Canto** and processed into a clean, uniform dataset suitable for deep learning.  
The final model uses **transfer learning with EfficientNetB0** to achieve strong performance on **28 bird species**.

---

## Problem Statement  
Classify bird sounds into 28 species using audio recordings collected in real-world conditions.

**Challenges:**
- Raw audio clips have **different lengths**, background noise, and silence  
- Recording quality varies significantly  
- Dataset size is limited  
- High similarity between bird calls across species  

---

## Data Collection (Xeno-Canto)

Bird audio recordings were sourced from **Xeno-Canto**, an open global repository of bird sounds.  
The recordings vary in:
- Duration (short clips to long recordings)  
- Background noise levels  
- Recording quality (A–D)  
- Presence or absence of bird calls  

An automated data collection script was used to download recordings for each bird species across multiple quality levels to improve robustness.

---

## Audio Preprocessing & Cleaning Pipeline

### Why this step was needed  
Deep learning models require **fixed-size inputs** and benefit from **clean signal-to-noise ratio**.  
Raw Xeno-Canto audio contains silence, noise, and uneven clip lengths.

### What was done  
- Standardized recordings into **fixed 10-second audio segments**  
- Automatically **filtered out low-energy / silent segments** using RMS energy thresholding  
- Sampled recordings across **multiple quality levels (A–D)** to improve noise robustness  
- Skipped corrupted or invalid audio files  

This ensured the dataset contains **actual bird vocalizations**, not silence or background noise.

---

## Feature Engineering: Mel Spectrograms

### Why spectrograms?  
Audio was converted into **Mel spectrogram images**, which:
- Capture time–frequency patterns of bird calls  
- Work well with CNNs pretrained on ImageNet  
- Enable reuse of strong vision backbones (EfficientNet)  

### What was done  
- Converted each 10-second audio chunk into a **Mel spectrogram**  
- Applied **log-scale (dB) normalization**  
- Exported spectrograms as **224×224 RGB images** for CNN input  

---

## Dataset Curation

- Species: **28 bird classes**  
- Final dataset size: **5,000+ spectrogram images**  
- Balanced sampling across classes  
- Controlled number of samples per recording to avoid dataset bias  
- Train / Validation / Test split: **80% / 10% / 10%**

---

## Model Architecture

- Backbone: **EfficientNetB0 (pretrained on ImageNet)**  
- Global Average Pooling  
- Dropout (0.3)  
- Dense Softmax classifier (28 classes)  
- L2 regularization on final layer  

---

## Training Strategy (Staged Transfer Learning)

### Stage 1 – Train Classification Head  
- Backbone frozen  
- Only classifier layers trained  
- Learning rate: 1e-4  

### Stage 2 – Fine-tuning  
- Unfroze top layers of EfficientNetB0  
- Lower learning rate: 1e-5  
- Adapted pretrained features to spectrogram-specific patterns  

---

## Results

| Metric                  | Value              |
|-------------------------|--------------------|
| Classes                 | 28                 |
| Final Validation Acc    | **85.5%**          |
| Final Test Accuracy     | **76.7%**          |
| Validation Loss         | **3.10 → 0.69**    |
| Validation Loss Drop    | **~78%**           |
| Dataset Expansion       | **5× (via aug.)**  |

Training curves showed steady improvement with no severe overfitting.

---

## Evaluation

- Evaluated on a held-out test set not seen during training  
- Validation and test gap reflects realistic generalization behavior  
- Confirms the model learned meaningful acoustic patterns rather than memorizing samples  

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- EfficientNetB0  
- Librosa  
- NumPy  
- Matplotlib  

---

## Key Learnings

- Built a **real-world audio preprocessing pipeline**  
- Applied **transfer learning** to non-vision data (audio → spectrograms)  
- Improved generalization using **data augmentation + regularization**  
- Designed a **staged fine-tuning strategy** to avoid overfitting  
- Interpreted training curves to debug model performance  

---

## Future Improvements

- Add Top-K accuracy for noisy audio  
- Confusion matrix analysis for error diagnosis  
- Optimize model for edge-device inference  
- Experiment with lightweight backbones (MobileNet)  

---

## Why This Project Matters

This project demonstrates end-to-end ML engineering:
- Data collection from noisy real-world sources  
- Signal processing → ML feature engineering  
- Model training and fine-tuning  
- Proper evaluation on unseen data  

It reflects practical skills used in production ML workflows, not just academic experiments.
