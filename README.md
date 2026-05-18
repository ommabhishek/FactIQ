# 📰 Fake News Detection — NLP-Based Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?style=flat-square&logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-3.x-green?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)
![NSoC](https://img.shields.io/badge/NSoC'26-Open%20Source-purple?style=flat-square)

> An end-to-end Natural Language Processing pipeline that classifies news articles as **real** or **fake** using classical ML techniques — from raw text to a trained Logistic Regression model.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Tech Stack](#-tech-stack)
- [Features & Workflow](#-features--workflow)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Overview

This project tackles the growing problem of misinformation by building a supervised machine learning classifier trained on labeled news datasets. The pipeline processes raw text through NLP preprocessing, converts it into numerical features via vectorization, and uses **Logistic Regression** to distinguish fake news from real news.

This repository was created as part of **NSoC'26** (Namespace Open Source Contribution Program).

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.8+** | Core programming language |
| **Pandas** | Data loading, cleaning, and manipulation |
| **NLTK** | Tokenization, stopword removal, stemming |
| **Scikit-Learn** | TF-IDF Vectorization + Logistic Regression |
| **Jupyter Notebook** | Interactive development and visualization |

---

## ✨ Features & Workflow

The project follows a clean, modular NLP pipeline:

```
Raw Text Input
     │
     ▼
① Text Preprocessing   ← Lowercasing, punctuation removal, stopwords, stemming (NLTK)
     │
     ▼
② Vectorization        ← TF-IDF transformation (Scikit-Learn)
     │
     ▼
③ Model Training       ← Logistic Regression classifier
     │
     ▼
④ Prediction           ← Real / Fake classification with confidence score
```

- ✅ Cleans and normalizes raw news text
- ✅ Removes noise (stopwords, special characters)
- ✅ Converts text to TF-IDF feature vectors
- ✅ Trains a Logistic Regression classifier
- ✅ Evaluates model with accuracy, precision, recall, and F1-score

---

## 📁 Project Structure

```
fake-news-detection/
│
├── data/
│   ├── train.csv          # Training dataset
│   └── test.csv           # Testing dataset
│
├── main.ipynb             # Main Jupyter Notebook (full pipeline)
│
├── requirements.txt       # Python dependencies
│
└── README.md              # You are here
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have **Python 3.8+** installed. You can check with:

```bash
python --version
```

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## ▶️ How to Run

### Option A — Jupyter Notebook (Recommended)

```bash
jupyter notebook main.ipynb
```

Open the notebook in your browser, then run all cells in order (`Kernel → Restart & Run All`).

### Option B — JupyterLab

```bash
jupyter lab main.ipynb
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | ~95% |
| Precision | ~94% |
| Recall | ~95% |
| F1-Score | ~94% |

> *Results may vary depending on the dataset used.*

---

## 🤝 Contributing

Contributions are welcome! This project is part of **NSoC'26**.

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes and commit: `git commit -m "Add: your feature"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request 🎉

Please make sure your code follows the existing style and includes comments where necessary.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ for NSoC'26 · Open Source</p>
