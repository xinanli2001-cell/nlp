"""Pre-download bert-base-uncased and roberta-base to local HF cache with retries."""
import time
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel

ATTEMPTS = 6
DELAY = 10


def try_download(fn, name: str):
    for i in range(ATTEMPTS):
        try:
            print(f"[{i+1}/{ATTEMPTS}] downloading {name} ...", flush=True)
            fn()
            print(f"  OK {name}", flush=True)
            return
        except Exception as e:
            print(f"  retry after {type(e).__name__}: {str(e)[:160]}", flush=True)
            time.sleep(DELAY)
    raise RuntimeError(f"exhausted retries for {name}")


try_download(lambda: BertTokenizer.from_pretrained("bert-base-uncased"), "bert tokenizer")
try_download(lambda: BertModel.from_pretrained("bert-base-uncased"), "bert-base-uncased model")
try_download(lambda: RobertaTokenizer.from_pretrained("roberta-base"), "roberta tokenizer")
try_download(lambda: RobertaModel.from_pretrained("roberta-base"), "roberta-base model")
print("ALL MODELS CACHED", flush=True)
