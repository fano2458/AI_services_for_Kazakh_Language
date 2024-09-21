import pandas as pd

splits = {'train': 'kazqad/kazqad-reading-comprehension-v1.0-kk-train.jsonl.gz', 'valid': 'kazqad/kazqad-reading-comprehension-v1.0-kk-validation.jsonl.gz', 'test': 'kazqad/kazqad-reading-comprehension-v1.0-kk-test.jsonl.gz'}

train_df = pd.read_json("hf://datasets/issai/kazqad/" + splits["train"], lines=True)
valid_df = pd.read_json("hf://datasets/issai/kazqad/" + splits["valid"], lines=True)
test_df = pd.read_json("hf://datasets/issai/kazqad/" + splits["test"], lines=True)

#  mBERT, XLM-R, XLM-V, Kaz-RoBERTa
print(train_df.columns)