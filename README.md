# 🧠 Drowsiness Detection Model - BiLSTM with Attention

## 📊 النتائج النهائية

### Test Set Performance
- **Accuracy:** 94.63%
- **Precision:** 93.54%
- **Recall:** 95.02%
- **F1-Score:** 94.27%
- **Specificity:** 94.29%

### Error Analysis
- True Positives (كشف صحيح للنعاس): 27,030
- True Negatives (كشف صحيح للاستيقاظ): 30,832
- False Positives (إنذارات كاذبة): 1,867
- False Negatives (حالات فاتة): 1,418

---

## 🏗️ معمارية النموذج

```
Input (Sequence)
    ↓
BiLSTM (Bidirectional LSTM - 128 hidden units, 3 layers)
    ↓
Multi-Head Attention (4 heads)
    ↓
Dense Layers (256 → 128 → 64 → 32 → 1)
    ↓
Sigmoid (Binary Classification)
    ↓
Output (Alert or Drowsy)
```

---

## 📁 الملفات المرفقة

1. **drowsiness_detector_final.pth** - النموذج المدرب (الأوزان)
2. **drowsiness_detector_info.json** - معلومات النموذج
3. **feature_scaler.pkl** - Scaler لتطبيع البيانات
4. **README.md** - هذا الملف

---

## 🚀 كيفية الاستخدام

### 1. التثبيت

```python
import torch
import pickle
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

# حمّل النموذج
checkpoint = torch.load('drowsiness_detector_final.pth')
model_state = checkpoint['model_state_dict']
features = checkpoint['features']
scaler_params = checkpoint['scaler_params']
sequence_length = checkpoint['sequence_length']
```

### 2. تحضير البيانات

```python
# استخدم نفس الميزات (22 feature)
X = your_data[features]  # (N, 22)

# تطبيع البيانات
scaler = StandardScaler()
scaler.mean_ = np.array(scaler_params['mean'])
scaler.scale_ = np.array(scaler_params['scale'])
X_scaled = scaler.transform(X)

# إنشاء تسلسلات
sequence = X_scaled[i:i+sequence_length]  # (15, 22)
```

### 3. التنبؤ

```python
import torch

# حوّل إلى tensor
X_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # (1, 15, 22)

# التنبؤ
with torch.no_grad():
    output = model(X_tensor)
    probability = torch.sigmoid(output).item()
    prediction = 'Drowsy' if probability > 0.5 else 'Alert'
    
print(f"Probability: {probability:.4f}")
print(f"Prediction: {prediction}")
```

---

## 📊 الميزات المستخدمة (22 Feature)

الميزات تتضمن:
- **Eye Features:** EAR (Eye Aspect Ratio), متوسط انغلاق العين
- **Mouth Features:** MAR (Mouth Aspect Ratio)
- **Gaze Features:** زوايا النظر (X, Y)
- **Head Pose:** دوران الرأس (Rx, Ry, Rz)
- والمزيد من الميزات الوجهية

---

## ⚙️ المتطلبات

```
PyTorch >= 1.9.0
NumPy >= 1.19.0
scikit-learn >= 0.24.0
```

---

## 🔍 ملاحظات مهمة

1. **الـ Sequence Length:** يجب أن تكون البيانات في تسلسلات بطول 15 frame
2. **معدل الإطارات:** النموذج مُدرّب على إطارات بتتابع معين
3. **الإضاءة:** أفضل الأداء مع إضاءة جيدة
4. **الزاوية:** أفضل الأداء عندما تكون الكاميرا مواجهة للوجه
5. **معايرة:** تأكد من تطبيع البيانات بنفس الـ scaler

---

## 📈 التحسينات المستقبلية

- [ ] إضافة ميزات زمنية (derivatives, velocity)
- [ ] دعم Real-time processing
- [ ] تحسين كشف النعاس في ظروف إضاءة منخفضة
- [ ] تقليل عدد الميزات المطلوبة

---

## 📧 الدعم

للمزيد من المعلومات أو المساعدة:
- راجع ملف `drowsiness_detector_info.json`
- تحقق من أن البيانات بالصيغة الصحيحة

---

## 📜 الترخيص

هذا النموذج متاح للاستخدام التعليمي والبحثي.

---

**تم إنشاء هذا النموذج بنجاح! 🎉**
