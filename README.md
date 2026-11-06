# Hate Speech Detection on Tweets using Fine-Tuned DistilBERT

Classifies tweets as **hate speech**, **offensive language**, or **neither**.

**Live demo: piyushMehJ/Hate-Speech-Distilbert**

## Results (same 80/20 test split for every model)

|Model|Accuracy|Macro F1|Hate-class F1|
|-|-|-|-|
|TF-IDF + Logistic Regression|0.878|0.731|0.418|
|Bi-LSTM (dropout, L2, early stopping)|0.889|0.725|0.398|
|**DistilBERT (fine-tuned)**|**0.907**|**0.779**|**0.500**|

## What I did

* Dataset: 24,783 labeled tweets; hate speech is only \~6%, so I used class weights.
* Cleaned the text, built 3 models, and compared them fairly on the same test data.
* Fine-tuned DistilBERT (PyTorch + Hugging Face).

## Limitations

* Hate speech is the hardest class (F1 0.50). The dataset's labels are noisy, so it
often confuses hate speech with plain offensive language.
* Trained on English tweets only. This is a prototype, not a moderation system.

## Run it

`pip install -r requirements.txt` then `python app.py`

Tech: Python, PyTorch, TensorFlow/Keras, Hugging Face Transformers, scikit-learn, spaCy, Gradio

