# SentimentForge: Sentiment Analysis on IMDB Reviews

A deep learning project that classifies IMDB movie reviews as **positive** or **negative**. It compares a classical baseline with sequence models, quantizes the best model for fast inference, and serves it through a Gradio web app.

## Results

Test set: 4,959 reviews.

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **GRU + Word2Vec** | **0.9006** | **0.9029** | 0.9609 |
| TF-IDF + Logistic Regression | 0.8994 | 0.9009 | 0.9631 |
| BiLSTM | 0.8851 | 0.8848 | 0.9540 |
| GRU | 0.8820 | 0.8867 | 0.9513 |
| LSTM | 0.8786 | 0.8832 | 0.9514 |

All models exceed the 85% accuracy target.

### Optimization

| Version | Accuracy | Size (MB) | ms per review |
|---|---|---|---|
| GRU + Word2Vec (Keras, float32) | 0.9006 | 9.91 | 15.79 |
| TFLite float32 | 0.9006 | 9.92 | 1.97 |
| **TFLite int8** | 0.9006 | **2.49** | **1.42** |

int8 quantization makes the model about 4x smaller with no loss in accuracy. The Gradio app uses this version.

## Dataset

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle). It has 50,000 reviews, which become 49,582 after removing duplicates, and the classes are balanced.

The dataset is not included in this repository. Download `IMDB Dataset.csv` from the link above.

## Project structure

```
.
├── IMDB_documented.ipynb   # Full notebook: EDA, training, evaluation, deployment
├── requirements.txt        # Python dependencies
└── README.md
```

Running the notebook also creates the saved models and files (`*.keras`, `*.tflite`, `tokenizer.pkl`, `tfidf_logreg.pkl`, `config.json`, `stop_words.json`).

## Pipeline

1. **Data understanding:** text length and class distribution, word clouds, most frequent words, n-grams.
2. **Preprocessing:** lowercasing, HTML tag removal, contraction fixing, special-character and stopword removal; Keras and BERT tokenizers explored; train/validation/test split.
3. **Feature representations:** TF-IDF and Word2Vec embeddings (128 dimensions, frozen).
4. **Models:** TF-IDF + Logistic Regression, LSTM, GRU, BiLSTM, GRU + Word2Vec, using dropout, the Adam optimizer, `ReduceLROnPlateau` and `EarlyStopping`.
5. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC curves, training curves, per-class report.
6. **Error analysis:** 493 mistakes out of 4,959. Long reviews that get truncated, sarcasm and mixed sentiment are the main causes.
7. **Optimization:** float16 and int8 embedding quantization, then TFLite float32 and int8 conversion.
8. **Deployment:** Gradio app that returns the predicted sentiment with confidence scores.
9. **Insights and recommendations:** use cases, limitations and next steps are at the end of the notebook.

## Setup

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

The notebook downloads the NLTK stopwords data on first run.

## Running the notebook

**On Google Colab (used for this project):**
1. Upload `IMDB Dataset.csv` to the session storage (`/content/`).
2. Upload and open `IMDB_documented.ipynb`.
3. Run all cells from top to bottom.

**Locally:**
1. Place `IMDB Dataset.csv` next to the notebook.
2. Change the path in the data-loading cell from `/content/IMDB Dataset.csv` to `IMDB Dataset.csv`.
3. Run `jupyter notebook` and run all cells.

The last cell starts the Gradio app. In Colab it prints a temporary public link.

## Limitations

- Sarcasm and mixed opinions are sometimes misclassified, occasionally with high confidence.
- The task is binary only, with no neutral class.
- The model is trained on movie reviews, so accuracy on other domains is likely lower without retraining.
- Only the first 250 words of each review are used.

## Future work

- Fine-tune a pretrained transformer such as DistilBERT.
- Try GloVe or FastText embeddings.
- Add a neutral class and test longer maximum review lengths.
