# NLU-ASSG1

---
# Sport vs Politics Text Classification

This repository contains a machine learning based text classification system
that classifies news documents into two categories: **Sports** and
**Politics**. The project focuses on classical machine learning techniques
and emphasizes experimental rigor, interpretability, and comparative
analysis.

---

## Project Overview

Text classification is a fundamental problem in Natural Language Processing
(NLP). In this project, a supervised learning approach is used to classify
documents based on their textual content. Instead of using deep learning
models, the project deliberately employs traditional machine learning
algorithms to better understand how feature representations and model
assumptions influence performance.

The classification pipeline includes:
- Data collection and preprocessing
- Feature extraction using TF-IDF
- Training and evaluation of multiple classifiers
- Quantitative comparison of model performance

---

## Dataset

The dataset is sourced from the **20 Newsgroups dataset**, a standard
benchmark corpus widely used in text classification research.

The following categories were selected and grouped into two classes:

### Sports
- `rec.sport.baseball`
- `rec.sport.hockey`

### Politics
- `talk.politics.guns`
- `talk.politics.mideast`

During preprocessing, document headers, footers, and quoted replies were
removed to reduce noise and ensure that models learn from the main textual
content.

---

## Feature Representation

Text documents are converted into numerical feature vectors using
**Term Frequency–Inverse Document Frequency (TF-IDF)**.

Key feature settings:
- Unigrams and bigrams
- English stopword removal
- Rare term filtering
- High-dimensional sparse representation (~50,000 features)

TF-IDF was chosen because it effectively emphasizes discriminative terms
while reducing the influence of commonly occurring words.

---

## Machine Learning Models

The following machine learning models were trained and evaluated using the
same feature representation and data split:

- **Multinomial Naive Bayes**
- **Logistic Regression**
- **Support Vector Machine (Linear SVM)**

This setup allows for a fair comparison of probabilistic, discriminative,
and margin-based classifiers.

---

## Experimental Setup

- Train–test split: 80–20 (stratified)
- Feature extraction fitted only on training data to avoid information
  leakage
- Fixed random seeds used to ensure reproducibility
- Evaluation metrics include accuracy, precision, recall, and F1-score

---

## Results

| Model | Accuracy |
|------|----------|
| Naive Bayes | 96.49% |
| Logistic Regression | 94.93% |
| SVM | 96.10% |

Naive Bayes achieved the highest accuracy, closely followed by SVM.
Logistic Regression also performed competitively, demonstrating the
effectiveness of classical machine learning approaches for text
classification.

---

## Limitations

The models rely on surface-level lexical features and do not capture deeper
semantic meaning. Documents containing mixed-domain content may be
misclassified. Performance may also degrade under domain shifts or changes
in vocabulary.



## How to Run

1. Open the notebook `Sport_vs_Politics_Classifier.ipynb`
2. Run all cells from top to bottom
3. The notebook will load the dataset, train models, and display results

The notebook is designed to run in **Google Colab** without additional
setup.

---

## Conclusion

This project demonstrates that traditional machine learning models, when
combined with appropriate feature representations such as TF-IDF, can
achieve strong performance on text classification tasks. The comparative
analysis provides insight into how different modeling assumptions affect
classification behavior in high-dimensional textual data.

