# SEMSOL — AI CV & Audio Suite

This repository bundles two projects:
- Gaze Estimation (Computer Vision) — folder: `Gaze Estimation/`
- STT‑NLP (Speech‑to‑Text) — folder: `STT-NLP/`

Quick links
- STT‑NLP end‑user guide: `STT-NLP/README.md`
- Gaze Estimation usage (below)

Repository layout
```
SEMSOL/
├── Gaze Estimation/           # CV: real-time gaze, engagement, blink
├── STT-NLP/                   # Audio: STT with CNN+BiLSTM+CTC, Streamlit demo
├── requirements.txt           # consolidated deps for both projects
└── .gitignore                 # ignores datasets/, artifacts/, weights cache, pyc
```

# Gaze and Engagement Estimation with Blink Detection

This project performs **real-time gaze estimation, engagement scoring, and eye-blink detection** using deep learning and computer vision.  
It supports webcam input or pre-recorded videos and integrates a rule-based engagement scorer with an optional blink detector.

---

## Features
✅ Real-time **Gaze Estimation** using deep learning  
✅ **Engagement Scoring** (4 levels: Highly Engaged → Disengaged)  
✅ Optional **Blink Detection** using facial landmarks  
✅ Works with webcam or video input  
✅ Switchable between **Gaze360** and **MPIIGaze** datasets  
✅ Supports multiple backbones: `resnet18`, `resnet34`, `mobilenetv2`, etc.

---

## 1. Environment Setup

First, create and activate your virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Then, install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Running the Program

### Run with Webcam
Use this command to open your webcam and start real-time inference:
```bash
python inference.py --model resnet34 --weight weights/resnet34.pt --dataset gaze360 --source 0 --enable-blink
```
- `--source 0` → uses the default camera  
- `--enable-blink` → activates blink detection  
- Press **Q** to quit the live window  

---

### Run with a Video File
To process a saved video instead:
```bash
python inference.py --model resnet34 --weight weights/resnet34.pt --dataset gaze360 --source assets/in_video.mp4 --enable-blink
```
- Make sure your video file is located in the `assets` folder (or specify a full path).

---

### Save Output Video
To save the processed video:
```bash
python inference.py --source 0 --output output.mp4 --enable-blink
```
The result will be saved in your working directory as **output.mp4**.

---

## 3. Datasets

You can train or fine-tune on supported datasets:

### Gaze360 (Default)
Default dataset for gaze estimation, covering a wide range of head poses.  
Directory structure:
```
datasets/gaze360/
 ├── Image/
 └── Label/
```

### MPIIGaze
To use the MPIIGaze dataset:
```bash
python inference.py --model resnet34 --weight weights/resnet34.pt --dataset mpiigaze --source 0 --enable-blink
```

---

## 4. Engagement Levels

| Level | Score Range | Description          |
|:------|:-------------|:--------------------|
| 1     | 75–100       | Highly Engaged      |
| 2     | 50–74        | Engaged             |
| 3     | 25–49        | Partially Engaged   |
| 4     | 0–24         | Disengaged          |

---

## 5. Blink Detection

Blink detection uses facial landmarks via `cvzone` to monitor the eye aspect ratio.

Blink counting logic:
- Detects vertical and horizontal eye distances  
- Computes a ratio to determine eye closure  
- Increments counter upon each blink  

Blink data integrates with engagement score for deeper behavioral analysis.

---

## 6. Performance Tips

- Use a **GPU** for real-time performance.  
- Reduce input resolution or disable blink detection for faster FPS.  
- `mobilenetv2` offers faster inference than `resnet` models.

---

## 7. Fine-Tuning (Optional)

You can fine-tune the model on another dataset:
```bash
python main.py --model resnet34 --dataset gaze360 --weight weights/resnet34.pt --epochs 5 --lr 1e-4
```

---

## 8. Example Outputs

- Live gaze visualization with bounding boxes  
- Engagement score overlay (0–100)  
- Blink counter display  

---

STT‑NLP Quickstart
```
# install once (from repo root)
pip install -r requirements.txt

# inference
----

# run the demo
cd STT-NLP
streamlit run app.py

# train (with checkpoints & auto-resume)
python -u main.py         # or: python -u run_trainer.py
```

---

## STT‑NLP (Speech‑to‑Text)

### 1. Environment Setup
```
pip install -r requirements.txt
```

### 2. Running
- Inference (download weights):
  ```bash

  ```
- Streamlit demo:
  ```bash
  cd STT-NLP
  streamlit run app.py
  ```
- Programmatic inference:
  ```python
  from src.pipeline.inference_pipeline import InferencePipeline
  pipe = InferencePipeline()
  print(pipe.predict("/path/to/audio.wav"))
  ```

### 3. Training (Checkpoints & Auto‑Resume)
- End‑to‑end (ingestion → transformation → training):
  ```bash
  cd STT-NLP
  python -u main.py
  ```
- Train only:
  ```bash
  python -u run_trainer.py
  ```
Checkpoints are saved under `STT-NLP/artifacts/model_trainer/checkpoints/` (epoch_N.pt and latest.pt). Training resumes from the last epoch automatically.

### 4. Data
- Place LibriSpeech under `STT-NLP/datasets/LibriSpeech/` (e.g., `train-clean-100/`, `test-clean/`).

### 5. Troubleshooting
- If you see shape or NaN CTC loss errors, lower LR (1e-4), reduce model size in `params.yaml`, and ensure `n_feats` matches MelSpectrogram `n_mels`.
- To start fresh, delete `STT-NLP/artifacts/model_trainer/checkpoints/latest.pt` and rerun.

---

## Author
**Mario Gamal Sobhy** 
AI Engineer | Computer Vision | Deep Learning 
[GitHub Repository](https://github.com/Mario-Gamal-Sobhy/SEMSOL)<br>
**Abdelrahman Gaber**
AI Engineer | Linux | NLP

---

## License
This project is open-source under the MIT License.