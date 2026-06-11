# Indonesian SMS Spam Detection: A Comparative Study of TF-IDF + SVM and IndoBERT (PyTorch)

An end-to-end Machine Learning and Deep Learning project to classify Indonesian SMS messages into three distinct categories: **Normal**, **Fraud (Penipuan)**, and **Promo**. This project serves as a benchmarking study evaluating the trade-offs between a traditional, lightweight ML pipeline and a state-of-the-art Pre-trained Language Model (Transformer).

---

## Business & Technical Use Case
In industries like Banking (e-commerce internal risk, transaction security) and FMCG/Retail, filtering unsolicited or fraudulent messages is crucial to protect user trust and secure financial ecosystems. 
* **Baseline Approach:** Leverages statistical features (TF-IDF) combined with Chi-Square selection and an SVM classifier. Highly cost-effective and low-latency.
* **Advanced Approach:** Utilizes Contextual Embeddings via **IndoBERT** (`indobenchmark/indobert-base-p2`) fine-tuned using PyTorch. Highly accurate in understanding nuance, slang, and context in text.

---

## Dataset Specifications
Source : https://raw.githubusercontent.com/ksnugroho/klasifikasi-spam-sms/master/data/dataset_sms_spam_v1.csv
The dataset consists of thousands of Indonesian short messages labeled into 3 classes:

0 (Normal): Regular person-to-person SMS.

1 (Fraud): Phishing, fake lottery winnings, cash-loan scams, etc.

2 (Promo): Official provider advertisements, restaurant discounts, etc.

Exploratory Data Analysis (EDA) highlights distinct keyword frequencies per class, visualized cleanly through custom-masked WordClouds in the baseline notebook.

## Project Pipeline

### 1. Traditional ML Pipeline (Baseline)
`Data Collection` ➔ `Text Pre-processing (Case Folding, Filtering, Stopwords, Stemming)` ➔ `Exploratory Data Analysis (WordCloud per label)` ➔ `Feature Extraction (TF-IDF)` ➔ `Feature Selection (Chi-Square)` ➔ `Classifier (Support Vector Machine)`

### 2. Advanced Deep Learning Pipeline (IndoBERT)
`Data Loading` ➔ `Tokenization & Formatting (AutoTokenizer)` ➔ `Contextual Embedding Extraction` ➔ `Fine-Tuning Architecture` ➔ `Feed-Forward Neural Network (Classification Layer)` ➔ `PyTorch Optimization Loop`

---
## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/okeypeeps/indonesian-sms-spam-detection.git
   cd indonesian-sms-spam-detection
```

2. Install dependencies:
   Make sure you are running a Python 3.12+ environment.
   ```bash
pip install -r requirements.txt
```

3. **Required packages include:**
   * `torch`
   * `transformers`
   * `sentencepiece`
   * `pandas`
   * `scikit-learn`
   * `matplotlib`
   * `wordcloud`

## Repository Structure

```text
indonesian-sms-spam-detection/
├── assets/
│   └── pipeline_bert.png          	# Architecture diagrams
├── notebooks/
│   ├── 01_baseline_tfidf_svm.ipynb 	# EDA, WordCloud, TF-IDF + Chi-Square + SVM Pipeline
│   └── 02_advanced_indobert.ipynb  	# PyTorch Fine-Tuning IndoBERT Pipeline
├── requirements.txt               	# Project dependencies
└── README.md                      	# This file
```
