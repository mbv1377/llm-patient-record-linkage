# Code Availability

Training and inference code for every model evaluated in *"Leveraging Open-Weight Language Models for Automated Patient Record Linkage"* (Beheshti et al., submitted to *BMC Medical Informatics and Decision Making*).

The record linkage pipeline has two independent stages, evaluated separately (see the Methods section of the paper): **blocking** (Experiment 1) and **matching** (Experiment 2). This repository contains the codes for both.

## Data availability

The underlying patient data contains Protected Health Information (PHI) and is **not included** in this repository, per the manuscript's Data Availability declaration. 

## Blocking model (Experiment 1)

Corresponds to Figure 1 in the paper.

| File | Description |
|---|---|
| `RoBERTa_Embedding_Finetunning.ipynb` | Fine-tunes RoBERTa as a sentence-embedding model (`sentence-transformers`, mean pooling, cosine similarity loss against the Overall Similarity Score) — Figure 1(a). |
| `RoBERTa_Embedding_Finetuned_Inference.py` | `CandidatePairGenerator` class: encodes both datasets with the fine-tuned embedding model, builds a FAISS index, and retrieves candidate pairs via k-NN search filtered by a cosine similarity threshold.

## Matching model (Experiment 2)

### RoBERTa classifier (encoder-based baseline)

| File | Description |
|---|---|
| `RoBERTa_Classification_Finetuning.ipynb` | Fine-tunes RoBERTa as a binary sequence classifier (Match / Non-Match) on serialized record pairs. |
| `RoBERTa_Classification_Finetuned_Inference.ipynb` | Runs the fine-tuned classifier on the test set and reports accuracy / classification report. |

### Generative LLMs

Fine-tuning and inference code for each open-weight generative LLM evaluated in the study:

| Model | Fine-Tuning | Fine-Tuned Inference | Zero-Shot Inference |
|---|:---:|:---:|:---:|
| Llama-3.2-3B | ✓ | ✓ | ✓ |
| Llama-3.1-8B | ✓ | ✓ | ✓ |
| Mistral-7B | ✓ | ✓ | ✓ |
| Mistral-Small-24B | ✓ | ✓ | ✓ |
| Gemma-3-27B | ✓ | ✓ | ✓ |
| Llama-3.3-70B | — | — | ✓ |
| DeepSeek-R1-Distill-Llama-70B | — | — | ✓ |

Llama-3.3-70B and DeepSeek-R1 were evaluated zero-shot only (no fine-tuning), per the paper's Methods and Table 3.

DeepSeek-R1-Distill-Llama-70B's notebook differs structurally from the others, since it is a reasoning model that emits a chain-of-thought trace (wrapped in `<think>...</think>`) before its final answer:
- Evaluated only on the challenging subset of the test set (0.65 < Overall Similarity Score < 1.0, N≈2,736), not the full test set, per §3.3 of the paper.
- Uses a much larger generation budget (`max_new_tokens=2048` vs. 4 for every other model) and a correspondingly larger `max_seq_length`, to allow the reasoning trace to complete.
- Answer extraction searches only the text *after* `</think>` for "Yes"/"No", rather than the whole decoded output, to avoid picking up an incidental mention of either word from within the reasoning trace itself.
- Uses its own prompt format (`<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜>`) rather than a chat template, since no Unsloth-registered chat template exists for this model.

## Requirements

- [`unsloth`](https://github.com/unslothai/unsloth) (generative LLM fine-tuning/inference)
- `transformers`, `trl`, `datasets`, `torch`
- `sentence-transformers`, `faiss` (blocking model)
- `pandas`, `numpy`, `scikit-learn`, `evaluate`, `tqdm`

