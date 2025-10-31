# STT‑NLP (Speech‑to‑Text)

Turn speech audio into text with a compact CNN+BiLSTM model trained using CTC. This project ships with:
- One‑click inference using prebuilt weights (no training required)
- Full training pipeline with checkpoints & auto‑resume
- Streamlit demo app

---

## What you can do
- Transcribe WAV audio files locally on CPU or GPU
- Train or fine‑tune on LibriSpeech subsets
- Resume training from the last completed epoch automatically

---

## Quickstart 
1) Install dependencies (from SEMSOL root or here):
```
pip install -r ../requirements.txt
```

2) Download prebuilt weights :
```
still not found ...
```
This will place files in:
- artifacts/model_trainer/model.pt
- artifacts/data_transformation/preprocessor.pkl

3) Run the Streamlit demo:
```
streamlit run app.py
```
Upload a WAV file and get the transcription.

Programmatic inference:
```python
from src.pipeline.inference_pipeline import InferencePipeline
pipe = InferencePipeline()
print(pipe.predict("/path/to/audio.wav"))
```

---

## Training (optional)
If you want to retrain or fine‑tune:

1) Prepare data
- Place LibriSpeech under `datasets/LibriSpeech/` (e.g., `train-clean-100/`, `test-clean/`)

2) Configure
- `config.yaml` controls paths/artifacts
- `params.yaml` controls model & training hyperparams (epochs, batch_size, rnn_dim, etc.)

3) Run training (auto‑resume + checkpoints):
```
python -u main.py           # end‑to‑end: ingestion → transformation → training
# or only training (expects processed CSVs/tensors):
python -u run_trainer.py
```
Checkpoints are saved to `artifacts/model_trainer/checkpoints/`:
- `epoch_{N}.pt` saved after each epoch
- `latest.pt` always points to the most recent epoch
On start, training resumes from `latest.pt` (so if you stop at epoch 2, it continues from 3).

Final model is saved to `artifacts/model_trainer/model.pt`.

---

## Faster training tips (CPU)
- Use a small subset: edit `artifacts/data_ingestion/train.csv` and `test.csv` to keep only the first few thousand rows
- In `params.yaml`: set `batch_size: 8–12`, `n_rnn_layers: 2–3`, `rnn_dim: 256–384`, `epochs: 3–5`
- Preprocessing already caches spectrograms; keep artifacts on SSD
- Use DataLoader workers ~ cores‑1 (default PyTorch picks a safe value)

On GPU (Colab): change runtime to GPU, then:
```
!pip install -r ../requirements.txt
!python -u main.py
```

---

## How it works
- Audio → Mel spectrogram (torchaudio: Resample → MelSpectrogram; optional SpecAugment)
- CNN downsamples time dimension and extracts features
- BiLSTM models context forward+backward
- CTC loss trains without frame‑level alignments

Key paths & files
- Input WAVs: `artifacts/data_ingestion/audio/{train,test}/...`
- Preprocessor: `artifacts/data_transformation/preprocessor.pkl`
- Processed CSVs: `artifacts/data_transformation/{train,test}_processed.csv`
- Spectrogram tensors: `artifacts/data_transformation/{train,test}/*.pt`
- Model: `artifacts/model_trainer/model.pt`
- Checkpoints: `artifacts/model_trainer/checkpoints/`

---

## Configuration
- `config.yaml` — data/artifacts locations
- `params.yaml` —
  - `model_trainer.n_cnn_layers`, `stride`, `n_feats`: CNN front‑end & feature dims
  - `model_trainer.n_rnn_layers`, `rnn_dim`, `dropout`: BiLSTM
  - `model_trainer.n_class`, `blank_index`: CTC alphabet size (29 incl. blank)
  - `epochs`, `batch_size`, `learning_rate`

Character map: `src/config/char_map.txt` (includes `' 0` and `<SPACE> 1`).

---

## CLI reference
- End‑to‑end training: `python -u main.py`
- Train only (resume supported): `python -u run_trainer.py`
- Weights downloader: `python weights/download_weights.py`
- Demo app: `streamlit run app.py`

---

## Troubleshooting
- “not enough values to unpack” when loading char map
  - Ensure first line of `src/config/char_map.txt` is `' 0` (apostrophe and an index)
- NaN/Inf loss (CTC)
  - Reduce LR (e.g., 1e‑4), reduce model size, or ensure transcripts aren’t longer than downsampled time
- Shape mismatch `input.size(-1) != input_size`
  - Ensure `params.yaml:n_feats` matches preprocessing (`MelSpectrogram(n_mels=...)`) and `stride/n_cnn_layers` are consistent
- Resume mismatch
  - Delete `artifacts/model_trainer/checkpoints/latest.pt` and rerun

---

## Project layout
- `main.py` — full pipeline entrypoint
- `run_trainer.py` — training only (with checkpoints/resume)
- `app.py` — Streamlit UI
- `src/components/` — ingestion, preprocessing, dataset, trainer
- `src/model/` — CNN, BiLSTM, SpeechToText
- `src/pipeline/` — training, inference, evaluation pipelines
- `weights/` — downloader + instructions (weights themselves are ignored)

---

## Credits
- LibriSpeech dataset (OpenSLR)
- PyTorch & torchaudio
