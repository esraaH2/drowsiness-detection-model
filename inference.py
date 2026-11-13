"""
Drowsiness Detection — inference with the trained BiLSTM + Multi-Head Attention model.

Usage:
    python inference.py            # runs a demo prediction on random data

    from inference import DrowsinessDetector
    detector = DrowsinessDetector()
    result = detector.predict(features)   # features: array of shape (15, 22)
"""

import os

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(HERE, "drowsiness_detector_final.pth")


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        b, t, d = x.shape
        q = self.query(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.fc_out(out)


class BiLSTMAttention(nn.Module):
    def __init__(self, input_size=22, hidden_size=128, num_layers=3,
                 dropout=0.3, num_heads=4):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0)
        d = hidden_size * 2
        self.attention = MultiHeadAttention(d, num_heads)
        self.fc1, self.bn1 = nn.Linear(d, 256), nn.BatchNorm1d(256)
        self.fc2, self.bn2 = nn.Linear(256, 128), nn.BatchNorm1d(128)
        self.fc3, self.bn3 = nn.Linear(128, 64), nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.bilstm(x)               # (B, T, 256)
        out = self.attention(out)             # (B, T, 256)
        out = out.mean(dim=1)                 # (B, 256)
        out = self.dropout(self.relu(self.bn1(self.fc1(out))))
        out = self.dropout(self.relu(self.bn2(self.fc2(out))))
        out = self.dropout(self.relu(self.bn3(self.fc3(out))))
        out = self.relu(self.fc4(out))
        return self.fc5(out)                  # logits, (B, 1)


class DrowsinessDetector:
    """Loads the trained model and scaler, and predicts Alert / Drowsy."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        arch = ckpt["model_architecture"]
        self.model = BiLSTMAttention(**arch).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

        self.features = ckpt["features"]
        self.seq_len = ckpt["sequence_length"]
        self.threshold = ckpt.get("threshold", 0.5)
        # Scaler parameters are stored in the checkpoint (same values as feature_scaler.pkl)
        self.mean = np.asarray(ckpt["scaler_params"]["mean"], dtype=np.float32)
        self.scale = np.asarray(ckpt["scaler_params"]["scale"], dtype=np.float32)

    def preprocess(self, features):
        x = np.asarray(features, dtype=np.float32)
        expected = (self.seq_len, len(self.features))
        if x.shape != expected:
            raise ValueError(f"Expected input of shape {expected}, got {x.shape}")
        return (x - self.mean) / self.scale

    @torch.no_grad()
    def predict(self, features, normalized=False):
        """
        features   : array (15, 22) of raw feature values, in the order of self.features
        normalized : set True if the features are already standardized
        """
        x = np.asarray(features, dtype=np.float32) if normalized else self.preprocess(features)
        t = torch.from_numpy(x).unsqueeze(0).to(self.device)
        prob = torch.sigmoid(self.model(t)).item()
        return {
            "prediction": "Drowsy" if prob >= self.threshold else "Alert",
            "probability": prob,
        }


if __name__ == "__main__":
    detector = DrowsinessDetector()
    print(f"Model loaded on {detector.device}")
    print(f"Input: {detector.seq_len} frames x {len(detector.features)} features")
    sample = detector.mean + detector.scale * np.random.randn(detector.seq_len, len(detector.features))
    result = detector.predict(sample)
    print(f"Prediction: {result['prediction']}  (probability {result['probability']:.4f})")
