# 📰 Fake News Detection using BiLSTM & GloVe

## 📌 Project Overview
This project focuses on detecting fake news using a deep learning model built with **Bidirectional LSTM (BiLSTM)** and **GloVe word embeddings**. The system processes the textual content of news articles and classifies them as *real* or *fake*.

## 🚀 Technologies Used
- Python
- TensorFlow / Keras
- GloVe (Pre-trained Word Embeddings)
- Scikit-learn
- Pandas / NumPy
- KerasTuner (for hyperparameter tuning)
- Matplotlib / Seaborn

## 📂 Dataset
Multiple datasets were combined from **Kaggle**:  
- LIAR Dataset  
- ISOT Fake News Dataset  
- WELFake Dataset

Final dataset contains: `title`, `text`, and `label` columns.

---

## 🛠 Preprocessing Steps
- Removed null/missing values
- Retained only `title`, `text`, and `label`
- Tokenization and lowercasing
- Stopword removal (NLTK)
- Lemmatization (WordNet)
- Token sequences padded to a fixed length

---

## 🧠 Model Architecture
- Embedding Layer (GloVe, trainable)
- Spatial Dropout
- Bidirectional LSTM
- Dropout
- Dense output layer with sigmoid activation

> Optimized via **KerasTuner** for:
> - LSTM units
> - Dropout rate
> - Number of layers
> - Embedding trainability

---

## 📊 Evaluation Metrics
- Accuracy (~90%)
- Confusion Matrix
- Precision, Recall, F1-score

---

## 📈 Results & Comparisons
| Model          | Accuracy |
|----------------|----------|
| SVM            | ~67%     |
| Decision Tree  | ~64%     |
| Logistic Reg.  | ~69%     |
| XGBoost        | ~71%     |
| ANN            | ~72%     |
| CNN            | ~74%     |
| RNN            | ~77%     |
| LSTM           | ~80%     |
| **BiLSTM (Ours)** | **~90%**     |

---

## 📁 Repository Structure
```
├── Google_Code_BiLSTM.ipynb     # Final model notebook
├── older_model_experiments.ipynb  # ML and ANN/CNN trials
├── glove.6B.100d.txt             # Pre-trained GloVe vectors
├── model_saved.h5                # Trained BiLSTM model
├── tokenizer.pickle              # Tokenizer object
├── /data                         # Combined dataset CSVs
├── requirements.txt              # Required packages
└── README.md                     # Project overview (this file)
```

---

## 📦 Setup Instructions
1. Clone the repo:
```bash
git clone https://github.com/yourusername/fake-news-bilstm.git
cd fake-news-bilstm
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the notebook in Jupyter or Colab.

---

## 📌 Future Enhancements
- Add attention mechanism (self-attention or transformer-based)
- Deploy as a web app (Flask/Streamlit)
- Integrate metadata (e.g., author, timestamp)
- Use Explainable AI tools (SHAP, LIME)

---

## 🙏 Acknowledgments
- GloVe embeddings by Stanford NLP
- Datasets sourced from Kaggle

---

## 📃 License
MIT License © 2025 [Your Name]

