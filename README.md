# Project 1. Seq2Seq Num2Text

Natural Language Processing (2026-2) — Project 1

A Seq2Seq (GRU encoder-decoder) translator that reads numbers between three representations:
**digits**, **English words**, and **Korean (hangul)**. One model is trained for each of the six directions.

| Direction | Example |
|---|---|
| num → en | `12345` → `twelve thousand three hundred forty five` |
| en → num | `twelve thousand three hundred forty five` → `12345` |
| num → ko | `12345` → `만 이천삼백사십오` |
| ko → num | `만 이천삼백사십오` → `12345` |
| en → ko | `twelve thousand three hundred forty five` → `만 이천삼백사십오` |
| ko → en | `만 이천삼백사십오` → `twelve thousand three hundred forty five` |

Digits are always handled as **strings** (never converted to `int`).

## Files

| File | Description |
|---|---|
| `project01/num2text.ipynb` | All code: data loading, vocabularies, model, training, evaluation, and the final load-and-test cell (with outputs kept) |
| `project01/num2text_models.pt` | Saved PyTorch checkpoint containing the config, the three vocabularies, and the weights of all six models |
| `project01/num2txt.txt` | Number ↔ English data (`number<TAB>english`, 100,000 lines) |
| `project01/num2kor.txt` | Number ↔ Korean data (`number<TAB>hangul`, 100,000 lines) |

## Data

- `num2txt.txt` and `num2kor.txt`, 100,000 pairs each, numbers from 0 up to 10 digits.
- English uses singular units (`thousand`, `million`, `billion`).
- Korean follows the format of the provided (revised) data: a space after each `억` / `만`
  group, and `일` is omitted before a unit (`만 이천삼백사십오`, not `일만이천삼백사십오`).
- The two files contain different random numbers; only **12,738** numbers appear in both,
  so the en ↔ ko pairs are built from that intersection.

### Split

Each direction is split 80 / 10 / 10 (train / val / test) with `random.seed(42)`.
The split is done independently per language pair, so **each direction has its own model**
(models are not shared across directions, which avoids test/train leakage between them).

| Pair | Train | Val | Test |
|---|---|---|---|
| num ↔ en | 80,000 | 10,000 | 10,000 |
| num ↔ ko | 80,000 | 10,000 | 10,000 |
| en ↔ ko | 10,190 | 1,273 | 1,275 |

## Method

**Tokenization**

- Numbers: one token per digit character.
- English: one token per word (split on spaces), 32 word types.
- Korean: one token per character, including the space (16 character types).
- Special tokens: `<pad>`=0, `<sos>`=1, `<eos>`=2, `<unk>`=3.

**Model** (Seq2Seq with GRU)

- Encoder: embedding (128) → 2-layer GRU (hidden 256, dropout 0.2); variable-length
  inputs are handled with `pack_padded_sequence`.
- Decoder: embedding (128) → 2-layer GRU (hidden 256, dropout 0.2) → linear layer over the target vocabulary.
  The encoder's final hidden state initializes the decoder.
- Decoding: **greedy decoding** (max length 30).

**Training**

- Loss: cross-entropy (padding ignored); optimizer: Adam, lr = 1e-3; batch size 128;
  gradient clipping at 1.0.
- Teacher forcing ratio 0.5 during training, 0 during validation / test.
- Reproducibility: the data split uses a fixed seed (`random.seed(42)`), but the PyTorch
  initialization seed is not fixed, so retraining can give slightly different accuracies.
- Epochs: 10 for the four num ↔ en / num ↔ ko directions, 30 for en ↔ ko
  (much less data).

## Results

Exact-match accuracy on the held-out test split (the whole output sequence must match).

| Direction | Test accuracy |
|---|---|
| num → en | 99.68 % (9968 / 10000) |
| en → num | 99.93 % (9993 / 10000) |
| num → ko | 99.77 % (9977 / 10000) |
| ko → num | 98.91 % (9891 / 10000) |
| en → ko | 99.37 % (1267 / 1275) |
| ko → en | 99.22 % (1265 / 1275) |

Typical errors are in the middle digits of long numbers (8–10 digits), e.g. swapping or
dropping a group such as `forty eight thousand` ↔ `forty four thousand`, or mis-handling
numbers with several zeros.

## How to use

The last cell of `num2text.ipynb` loads `num2text_models.pt` and runs the test code.
It defines its own copy of the vocabulary and model classes, so it can be run on its own
after a kernel restart (it only needs `num2text_models.pt` in the same directory).

```python
# last cell of num2text.ipynb
num2en("12345")                                    # 'twelve thousand three hundred forty five'
en2num("twelve thousand three hundred forty five") # '12345'
num2ko("12345")                                    # '만 이천삼백사십오'
ko2num("만 이천삼백사십오")                          # '12345'
en2ko("twelve thousand three hundred forty five")  # '만 이천삼백사십오'
ko2en("만 이천삼백사십오")                           # 'twelve thousand three hundred forty five'
```

Notes:

- Pass digits as a **string** (`"12345"`, not `12345`).
- English input uses singular units and single spaces between words.
- Korean input follows the data format (space after `억` / `만`).
- Numbers up to 10 digits are supported; longer numbers were not seen during training.

Run the notebook from the `project01` directory so that the data files and the checkpoint are found.
To retrain from scratch, run the notebook from top to bottom (a CUDA GPU is recommended,
CPU also works but is much slower).

## Environment

- Python 3.12, PyTorch 2.x (developed with 2.11.0 + CUDA 12.8 on an NVIDIA RTX GPU), NumPy.

## Team

- Team leader: `<GitHub ID>`
- Members: `<names>`