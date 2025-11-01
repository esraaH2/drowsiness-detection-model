# 🧠 Drowsiness Detection Model - BiLSTM with Attention

A deep learning model for detecting driver drowsiness using BiLSTM with Multi-Head Attention mechanism.

## 📊 Final Results

![Test Results](final_test_results.png)

### Performance Metrics

- **Accuracy:** 94.63%
- **Precision:** 93.54%
- **Recall:** 95.02%
- **F1-Score:** 94.27%
- **Specificity:** 94.29%

### Error Analysis

- **True Positives (Correct Drowsy Detection):** 27,030
- **True Negatives (Correct Alert Detection):** 30,832
- **False Positives (False Alarms):** 1,867
- **False Negatives (Missed Detections):** 1,418

## 🏗️ Model Architecture
```
Input (Sequence: 15 frames, 22 features)
    ↓
BiLSTM (128 hidden units, 3 layers, bidirectional)
    ↓
Multi-Head Attention (4 heads)
    ↓
Dense Layers (256 → 128 → 64 → 32 → 1)
    ↓
Sigmoid Activation
    ↓
Output (Alert or Drowsy)
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Usage
```python
from inference import DrowsinessDetector
import numpy as np

# Load the model
detector = DrowsinessDetector()

# Sample data (15 frames, 22 features)
features = np.random.randn(15, 22)

# Make prediction
result = detector.predict(features)
print(f"Prediction: {result['prediction']}")      # Alert or Drowsy
print(f"Probability: {result['probability']:.4f}") # 0.0-1.0
```

## 📋 Requirements
```
torch>=1.9.0
numpy>=1.19.0
scikit-learn>=0.24.0
pandas>=1.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## 📁 Project Structure
```
drowsiness-detection-model/
├── drowsiness_detector_final.pth      # Trained model weights
├── drowsiness_detector_info.json      # Model information
├── feature_scaler.pkl                 # Data scaler
├── inference.py                       # Example usage
├── requirements.txt                   # Dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

## 🔬 Features & Data

### Input Features (22 total)

- **Eye Features:** Eye Aspect Ratio (EAR), eye closure speed
- **Mouth Features:** Mouth Aspect Ratio (MAR)
- **Gaze Features:** Gaze angles (X, Y)
- **Head Pose:** Head rotation (Rx, Ry, Rz)
- **Additional:** Facial landmarks and temporal features

### Sequence Configuration

- **Sequence Length:** 15 frames
- **Frame Rate:** Real-time processing capable
- **Input Shape:** (15, 22)

## 📊 Model Details

- **Architecture:** BiLSTM + Multi-Head Attention
- **Hidden Size:** 128
- **Number of Layers:** 3
- **Attention Heads:** 4
- **Dropout:** 0.3
- **Loss Function:** Focal Loss
- **Optimizer:** AdamW
- **Scheduler:** CosineAnnealingWarmRestarts

## 🎯 Performance Analysis

### Test Results
- No overfitting detected (Val F1 ≈ Test F1)
- Excellent generalization
- Production-ready

### Confusion Matrix Interpretation
- **High True Positives:** Excellent drowsiness detection (95.02% recall)
- **High True Negatives:** Reliable alert state detection (94.29% specificity)
- **Low False Positives:** Minimal false alarms (5.7%)
- **Low False Negatives:** Rare missed detections (4.98%)

## 💡 Usage Recommendations

1. ✅ Ensure input data is properly normalized using the provided scaler
2. ✅ Use sequences of exactly 15 frames
3. ✅ Best performance with frontal face view
4. ✅ Works well in adequate lighting conditions
5. ✅ Real-time inference capable on CPU/GPU

## 🔄 Data Processing Pipeline
```python
1. Extract 22 facial features
2. Create sequences of 15 frames
3. Normalize using feature_scaler.pkl
4. Pass through BiLSTM model
5. Get probability output
6. Apply threshold (0.5)
7. Output prediction (Alert/Drowsy)
```

## 📈 Future Improvements

- [ ] Add temporal derivative features
- [ ] Optimize for edge devices (mobile)
- [ ] Improve low-light performance
- [ ] Reduce number of required features
- [ ] Real-time video processing pipeline
