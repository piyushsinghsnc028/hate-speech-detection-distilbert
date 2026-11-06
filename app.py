import re
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "piyushMehJ/Hate-Speech-Distilbert"  
LABELS = ["Hate speech", "Offensive language", "Neither"]

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()


def light_clean(s):
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"@\w+", " @user ", s)
    s = re.sub(r"&amp;|&#\d+;", " ", s)
    s = re.sub(r"\brt\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def classify(text):
    if not text or not text.strip():
        return {}
    enc = tok(light_clean(text), truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
    return {LABELS[i]: p for i, p in enumerate(probs)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, label="Type or paste a tweet"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    examples=[["Have a great day everyone!"],
              ["This traffic is so annoying, what a stupid road"]],
    title="Hate Speech Detection (fine-tuned DistilBERT)",
    description=("Classifies a tweet as hate speech, offensive language, or neither. "
                 "Prototype trained on a public, noisy-labeled tweet dataset "
                 "(hate-class F1 about 0.50), so it can make mistakes."),
)

if __name__ == "__main__":
    demo.launch()
