# ❤️ ECG Heartbeat Classification using Simple RNN

An end-to-end Deep Learning project that classifies ECG (Electrocardiogram) heartbeat signals using a **Simple Recurrent Neural Network (RNN)** built with **TensorFlow/Keras**. The project also includes an interactive **Streamlit** web application that allows users to classify ECG heartbeat signals in real time.

---

## 📌 Project Overview

Electrocardiograms (ECGs) are widely used to monitor heart activity and detect abnormalities. Manual analysis of ECG signals can be time-consuming and requires medical expertise.

This project demonstrates how a **One-to-One Recurrent Neural Network (RNN)** can automatically classify ECG heartbeat signals into different heartbeat categories using sequential time-series data.

The application accepts an ECG signal consisting of **140 sequential values**, processes the input using a trained RNN model, and predicts the heartbeat class along with the model's confidence score.

---

## 🚀 Features

- ❤️ One-to-One RNN architecture for ECG classification
- 📊 ECG5000 Dataset
- 🧠 TensorFlow/Keras Simple RNN model
- ⚡ Streamlit interactive web interface
- 📈 Displays prediction confidence
- 💾 Saves trained model and preprocessing objects
- 🔄 Standardized input using StandardScaler
- 🏥 Classifies multiple heartbeat types

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Joblib

---

## 📂 Project Structure

```
ECG-RNN/
│
├── app.py                  # Streamlit application
├── train.py                # Model training script
├── ecg.csv                 # ECG dataset
├── model.h5                # Trained RNN model
├── scaler.pkl              # Feature scaler
├── label_encoder.pkl       # Label encoder
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The project uses the **ECG5000 Dataset**, which contains ECG heartbeat recordings.

### Dataset Information

- Total Samples: ~5000
- Features: 140 ECG signal values
- Target: Heartbeat Class
- Data Type: Time-Series
- Learning Type: Multi-Class Classification

Each ECG sample consists of **140 sequential measurements** representing one heartbeat.

---

## ❤️ Heartbeat Classes

| Class | Heartbeat Type |
|--------|----------------|
| 1 | Normal Heartbeat |
| 2 | Supraventricular Premature Beat (SVPB) |
| 3 | Premature Ventricular Contraction (PVC) |
| 4 | Fusion Beat |
| 5 | Unknown / Unclassifiable Beat |

> **Note:** During model training, the labels are encoded from **1–5** to **0–4** using `LabelEncoder`. The Streamlit application converts the predicted labels back to their original heartbeat classes before displaying the result.

---

## 🧠 Model Architecture

```
Input (140 Timesteps)
          │
          ▼
Simple RNN (64 Units)
          │
          ▼
Dense Layer (32 Units)
          │
          ▼
Softmax Output Layer
          │
          ▼
Predicted Heartbeat Class
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/ECG-RNN.git
```

Move into the project directory

```bash
cd ECG-RNN
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

Run

```bash
python train.py
```

After training, the following files will be generated:

```
model.h5
scaler.pkl
label_encoder.pkl
```

---

## ▶️ Run the Streamlit Application

```bash
streamlit run app.py
```

The application will automatically open in your browser.

---

## 📱 Application Workflow

1. Load trained model.
2. Enter **140 ECG values** separated by commas.
3. Click **Predict**.
4. The model preprocesses the input.
5. The ECG signal is classified.
6. The predicted heartbeat type and confidence score are displayed.

---

## 📈 Data Preprocessing

- Load ECG dataset
- Separate features and labels
- Encode labels using LabelEncoder
- Standardize features using StandardScaler
- Reshape data to

```
(samples, 140, 1)
```

for RNN input.

---

## 📊 Model Training

- Loss Function:
  - Categorical Crossentropy

- Optimizer:
  - Adam

- Activation:
  - tanh (SimpleRNN)
  - ReLU (Dense)
  - Softmax (Output)

- Validation Split:
  - 20%

- Epochs:
  - 20

- Batch Size:
  - 32

---

## 🎯 Prediction Output

The application displays:

- Predicted Heartbeat Type
- Prediction Confidence
- ECG5000 Class Information

Example

```
Prediction:
❤️ Normal Heartbeat

Confidence:
98.80%
```

---

## 📚 Learning Objectives

This project demonstrates:

- Recurrent Neural Networks (RNN)
- One-to-One RNN Architecture
- Time-Series Classification
- ECG Signal Processing
- Feature Scaling
- Multi-Class Classification
- Deep Learning Model Deployment
- Streamlit Application Development

---

## 🔮 Future Improvements

- Upload ECG CSV files directly
- Display ECG waveform visualization
- Animated heartbeat background
- Model performance dashboard
- Probability chart for all classes
- LSTM and GRU model comparison
- Attention-based RNN implementation

---

## 👨‍💻 Author

**Arnav Misra**

Aspiring Data Scientist | Machine Learning Enthusiast | Deep Learning Developer

---

## ⭐ If you found this project useful

Give this repository a ⭐ on GitHub and feel free to contribute!