# ME-Whisper: Fine-tuning Whisper for Mandarin-English Code-Switching ASR

This repository contains the code for my MSc thesis at the University of Groningen (2023–2024), which investigates fine-tuning OpenAI's Whisper model for Mandarin-English intra-sentential code-switching speech recognition.

## Background

Code-switching (CS) — alternating between two languages within a single conversation — is common in bilingual communities but poses a significant challenge for standard ASR systems. This project addresses Mandarin-English CS, a particularly complex case due to the stark phonetic, syntactic, and semantic differences between the two languages.

## Dataset

- **Source:** TAL audio dataset of adult Mandarin-English mixed-language lectures
- **Speakers:** 200 speakers
- **Format:** 16kHz, 16-bit WAV
- **Splits used:** Development and test sets (training set excluded due to GPU/time constraints)

## Models

Two Whisper variants were fine-tuned:

| Model | Description |
|---|---|
| `ME-Whisper-small` | Fine-tuned from `whisper-small` |
| `ME-Whisper-large-v3` | Fine-tuned from `whisper-large-v3` |

## Results

Performance is measured using **Mixture Error Rate (MER)**, the standard metric for code-switching ASR evaluation.

| Experiment | Model | Test MER |
|---|---|---|
| Baseline | Whisper-small | 66.80% |
| Baseline | Whisper-large-v3 | 55.14% |
| Fine-tuned | ME-Whisper-small | **45.52%** |
| Fine-tuned | ME-Whisper-large-v3 | **39.07%** |

Fine-tuning reduced MER by **21.3 percentage points** on the small model and **16.1 percentage points** on the large model, demonstrating that targeted fine-tuning on a domain-specific CS dataset substantially improves recognition accuracy for bilingual speech.

## Requirements

```
torch
transformers
datasets
librosa
evaluate
jiwer
```


## Thesis

*Improving the Performance of Chinese-English Code-Switching Recognition Using Whisper*
University of Groningen, MSc Voice Technology, 2024

## Limitations

- Dataset size is modest; a larger and more diverse CS corpus would likely yield further improvements
- Hyperparameter tuning was limited by GPU availability and time constraints

## Future Work

- Expanding the dataset with more diverse accents and CS patterns
- Exploring cross-lingual transfer learning across other language pairs
- Developing severity-dependent models that adapt to varying degrees of code-switching
