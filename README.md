# imdb-sentiment-rnn

Binary sentiment classification of IMDb movie reviews (positive / negative) with recurrent neural networks in PyTorch. Text preprocessing and tokenization, Word2Vec embeddings, LSTM and GRU architectures, evaluated with precision, recall and F1-score.

> Mini-project 2 of the course *Técnicas de Deep Learning* (MAIA-4213), MSc in Artificial Intelligence, Universidad de los Andes.

## Problem

Given the text of a movie review, predict whether its sentiment is **positive** or **negative**. Recurrent networks read the review word by word and keep a hidden state that summarizes the context, which lets them capture negations and long-range cues ("not bad at all", "I expected more, but...").

**Dataset:** [IMDB Movie Ratings Sentiment Analysis, Kaggle](https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis/data). 40,000 reviews (`text`, `label`; 0 = negative, 1 = positive), versioned in `data/raw/movie.csv`.

## Approach

1. **Preprocessing:** HTML cleanup, lowercasing and tokenization that keeps negations and sentiment punctuation; removal of duplicated reviews before splitting; stratified train / validation / test split.
2. **Embeddings:** Word2Vec trained on the training reviews only, compared against randomly initialized embeddings.
3. **Recurrent models:** embedding layer, bidirectional LSTM or GRU, dropout and a dense output layer with sigmoid activation.
4. **Evaluation:** precision, recall, F1-score, accuracy, ROC-AUC and confusion matrix on the test set; mean ± std over seeds; comparison with a TF-IDF + logistic regression baseline; qualitative analysis of correct and misclassified reviews.

## Repository structure

```
imdb-sentiment-rnn/
├── data/
│   ├── raw/          # movie.csv, original Kaggle file
│   └── splits/       # train / val / test ids (versioned for reproducibility)
├── notebooks/        # solution notebook
├── outputs/
│   ├── figures/      # figures
│   ├── tables/       # metrics, histories and test predictions (CSV / JSON)
│   └── models/       # trained weights (.pt, not versioned)
├── report/           # IEEE / CVPR-style report (LaTeX) and its figures
├── docs/             # assignment statement and course report templates
└── requirements.txt
```

## How to run

```bash
git clone https://github.com/panamiguell/imdb-sentiment-rnn.git
cd imdb-sentiment-rnn
pip install -r requirements.txt jupyter
jupyter notebook notebooks/01_solucion_rnn_imdb.ipynb   # or open it in VS Code
```

Run all cells. Each experiment saves its results in `outputs/tables/`; if they already exist the notebook reloads them instead of retraining (set `RETRAIN = True` to train from scratch). The notebook runs on CPU, CUDA or Apple MPS.

## Results

_To be completed._
