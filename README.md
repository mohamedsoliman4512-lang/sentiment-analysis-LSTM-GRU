# 🧠 Sentiment Analysis using LSTM, GRU, and Bidirectional LSTM

A deep learning project that compares three RNN architectures for **multiclass sentiment classification** on text data.

---

## 📌 Project Overview

This project builds and evaluates three recurrent neural network models to classify text sentiment into **3 classes** (Positive / Negative / Neutral). The models are trained on a real-world sentiment dataset and compared across key performance metrics.

| Model | Architecture |
|-------|-------------|
| LSTM | Long Short-Term Memory |
| GRU | Gated Recurrent Unit |
| BiLSTM | Bidirectional LSTM |

---

## 🏆 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **GRU** | ✅ 0.845972 | ✅ 0.847144 | ✅ 0.845972 | **84.61%** |
| BiLSTM | 0.839954 | 0.839966 | 0.839954 | 0.839369 |
|  LSTM | 0.829536 | 0.830912 |0.829536 | 0.829970 |

> **GRU outperformed both LSTM and BiLSTM**, achieving the highest F1-Score of **84.61%** while requiring fewer parameters and faster training time.

---

## 🗂️ Dataset

- **Source:** [Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abdelmalekeladjelet/sentiment-analysis-dataset) (Kaggle)
- **File:** `sentiment_data.csv`
- **Target Column:** `Sentiment` (3 classes)
- **Text Column:** `Comment`

---

## ⚙️ Pipeline

```
Raw Text → Cleaning → Tokenization → Padding → Model Training → Evaluation
```

1. **Preprocessing** — Remove URLs, mentions, extra whitespace
2. **Tokenization** — Keras Tokenizer (vocab size: 50,000)
3. **Padding** — Max sequence length: 40 tokens
4. **Modeling** — LSTM / GRU / BiLSTM with Embedding + Dropout layers
5. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix

---

## 🧱 Model Architecture (shared across all three)

```
Embedding(50000, 100)
↓
SpatialDropout1D(0.2)
↓
LSTM / GRU / Bidirectional(LSTM) — 128 units
↓
Dropout(0.3)
↓
Dense(64, relu)
↓
Dense(3, softmax)
```

**Callbacks used:** EarlyStopping · ReduceLROnPlateau · ModelCheckpoint

---

## 🔧 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-red?logo=keras)
![Pandas](https://img.shields.io/badge/Pandas-purple?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn)

- Python 3.10+
- TensorFlow / Keras
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/mohamedsoliman4512-lang/sentiment-analysis-LSTM-GRU.git
cd sentiment-analysis-LSTM-GRU

# 2. Install dependencies
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn kagglehub

# 3. Open the notebook
jupyter notebook _Sentiment_Analysis_using_LSTM__GRU_and_Bidirectional_LSTM.ipynb
```

> The dataset is downloaded automatically via `kagglehub` — no manual download needed.

---

## 🔮 Future Work

- [ ] Pretrained word embeddings (GloVe / FastText)
- [ ] Transformer-based models (BERT, AraBERT)
- [ ] Hyperparameter optimization (Optuna / Keras Tuner)
- [ ] Attention mechanisms

---

## 👤 Author

**Mohamed Emad**  
Data Scientist & AI/ML Engineer  
[![LinkedIn](www.linkedin.com/in/mohamed-soliman4512)
[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/mohamedsoliman4512-lang)
[![Kaggle](https://www.kaggle.com/mohamedsoliman22))
