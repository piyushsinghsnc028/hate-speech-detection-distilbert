# Hate Speech Detection on Tweets using Fine-Tuned DistilBERT

Classifies a tweet as **hate speech**, **offensive language**, or **neither**.

**Model:** [Hugging Face](https://huggingface.co/piyushMehJ/Hate-Speech-Distilbert)

**Run it in Colab:** [Open in Colab](https://colab.research.google.com/drive/1S9Z2Mj4tVlOgDo8S2RLqu9lWGW5aUI68?usp=sharing)

## Results

All three models use the same 80/20 train/test split and the same metrics.

| Model | Accuracy | Macro F1 | Hate-class F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.878 | 0.731 | 0.418 |
| Bi-LSTM (dropout, L2, early stopping) | 0.889 | 0.725 | 0.398 |
| **DistilBERT (fine-tuned)** | **0.907** | **0.779** | **0.500** |

DistilBERT also reached a weighted F1 of 0.910, with hate-class precision 0.455 and recall 0.555.

## What I did

- **Data:** 24,783 labeled tweets in 3 classes. Hate speech is only about 6% of the data, so I used class weights so the model does not ignore it.
- **Cleaning:** removed URLs and mentions (regex); used spaCy lemmatization for the LSTM.
- **Models:** built three models and compared them fairly on the same test data: TF-IDF + logistic regression, a regularized Bi-LSTM, and fine-tuned DistilBERT.
- **Overfitting:** the first LSTM scored 97% on training data but only about 83% on test data. Adding dropout, L2 regularization and early stopping cut that gap from 14 to 4.5 points.
- **Final model:** fine-tuned DistilBERT with PyTorch and Hugging Face Transformers, keeping the best epoch based on a held-out validation set.

## Limitations

- Hate speech is the hardest class (F1 0.50). The dataset's labels are noisy, so the model often confuses hate speech with plain offensive language.
- English tweets only. This is a prototype, not a moderation system.

## Run it yourself

**In Colab (easiest):** open the Colab link above, switch to a GPU runtime (`Runtime > Change runtime type > T4 GPU`) and run the cells in order.

**Demo on your own computer:**
1. `pip install -r requirements.txt`
2. In `app.py`, make sure `MODEL_ID` is `piyushMehJ/Hate-Speech-Distilbert`.
3. `python app.py` and open the link it prints.

## Files

- `app.py`: Gradio demo that loads the model from Hugging Face
- `requirements.txt`: packages for the demo
- the notebook: data cleaning, all three models, and the results table

**Tech:** Python, PyTorch, TensorFlow/Keras, Hugging Face Transformers, scikit-learn, spaCy, Gradio
