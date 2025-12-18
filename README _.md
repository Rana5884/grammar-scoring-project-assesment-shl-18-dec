# 🎙️ Grammar Scoring Engine for Spoken Audio

## 📌 Project Overview

This project implements a **Grammar Scoring Engine** that predicts a
**continuous MOS Likert Grammar Score (0--5)** from spoken English audio
samples. Each audio file is 45--60 seconds long. The solution uses
**audio-based feature extraction** with Librosa and a **machine learning
regression model** to estimate grammatical quality from speech.

This project was developed as part of the **SHL Intern Hiring Assessment
2025** Kaggle competition.

------------------------------------------------------------------------

## 🧠 Problem Statement

Given an audio recording of spoken English, predict a **grammar quality
score** ranging from **0 (poor grammar)** to **5 (excellent grammar)**.\
This is modeled as a **supervised regression problem**.

------------------------------------------------------------------------

## 📂 Dataset Structure

    dataset/
    │
    ├── audios/
    │   ├── train/   → Training audio files (.wav)
    │   └── test/    → Test audio files (.wav)
    │
    ├── csvs/
    │   ├── train.csv → filename, label
    │   └── test.csv  → filename

-   **Training samples:** 409\
-   **Testing samples:** 197\
-   **Labels:** Continuous MOS grammar scores (0--5)

------------------------------------------------------------------------

## ⚙️ Feature Engineering

Audio features are extracted using **Librosa** and converted into
fixed-length vectors:

-   **MFCCs (13 coefficients)**
    -   Mean and standard deviation (26 features)
-   **Spectral features**
    -   Spectral centroid
    -   Spectral rolloff
-   **Prosodic features**
    -   RMS energy
    -   Zero-crossing rate
    -   Tempo
    -   Audio duration

➡️ **Total features per sample: 32**

------------------------------------------------------------------------

## 🤖 Model Used

-   **Algorithm:** XGBoost Regressor
-   **Objective:** `reg:squarederror`

**Why XGBoost?** - Works well with small datasets - Handles non-linear
relationships - Robust to noisy features

------------------------------------------------------------------------

## 📈 Evaluation

-   **Metric:** Root Mean Squared Error (RMSE)
-   Predictions are **clipped between 0 and 5** to match the MOS scoring
    scale.

------------------------------------------------------------------------

## 📤 Submission Format

The final submission file follows the Kaggle-required format:

``` csv
filename,label
audio_401,3.42
audio_402,4.01
```

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   Python
-   Librosa
-   NumPy
-   Pandas
-   Scikit-learn
-   XGBoost
-   Kaggle Notebook Environment

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Integrate **ASR (speech-to-text)** models (Whisper / Wav2Vec2)
-   Add **NLP-based grammar features**
-   Combine **audio + text features**
-   Perform hyperparameter tuning and cross-validation

------------------------------------------------------------------------

## 🧑‍💻 Author

**Gajender Chauhan**\
Data Operations Analyst \| Aspiring Machine Learning Engineer
