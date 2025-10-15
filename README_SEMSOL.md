# 👁️ SEMSOL – Gaze and Engagement Estimation with Blink Detection

This project performs **real-time gaze estimation, engagement scoring, and eye-blink detection** using deep learning and computer vision.  
It supports webcam input or pre-recorded videos and integrates a rule-based engagement scorer with an optional blink detector.

---

## 📦 Features
✅ Real-time **Gaze Estimation** using deep learning  
✅ **Engagement Scoring** (4 levels: Highly Engaged → Disengaged)  
✅ Optional **Blink Detection** using facial landmarks  
✅ Works with webcam or video input  
✅ Switchable between **Gaze360** and **MPIIGaze** datasets  
✅ Supports multiple backbones: `resnet18`, `resnet34`, `mobilenetv2`, etc.

---

## 🧰 1. Environment Setup

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

## 🧠 2. Running the Program

### ▶️ Run with Webcam
Use this command to open your webcam and start real-time inference:
```bash
python inference.py --model resnet34 --weight weights/resnet34.pt --dataset gaze360 --source 0 --enable-blink
```
- `--source 0` → uses the default camera  
- `--enable-blink` → activates blink detection  
- Press **Q** to quit the live window  

---

### ▶️ Run with a Video File
To process a saved video instead:
```bash
python inference.py --model resnet34 --weight weights/resnet34.pt --dataset gaze360 --source assets/in_video.mp4 --enable-blink
```
- Make sure your video file is located in the `assets` folder (or specify a full path).

---

### ▶️ Save Output Video
To save the processed video:
```bash
python inference.py --source 0 --output output.mp4 --enable-blink
```
The result will be saved in your working directory as **output.mp4**.

---

## 🧩 3. Datasets

You can train or fine-tune on supported datasets:

### 🔹 Gaze360 (Default)
Default dataset for gaze estimation, covering a wide range of head poses.  
Directory structure:
```
datasets/gaze360/
 ├── Image/
 └── Label/
```

### 🔹 MPIIGaze
To use the MPIIGaze dataset:
```bash
python inference.py --model resnet34 --weight weights/resnet34.pt --dataset mpiigaze --source 0 --enable-blink
```

---

## 🧮 4. Engagement Levels

| Level | Score Range | Description          |
|:------|:-------------|:--------------------|
| 1     | 75–100       | Highly Engaged      |
| 2     | 50–74        | Engaged             |
| 3     | 25–49        | Partially Engaged   |
| 4     | 0–24         | Disengaged          |

---

## 👁️ 5. Blink Detection

Blink detection uses facial landmarks via `cvzone` to monitor the eye aspect ratio.

Blink counting logic:
- Detects vertical and horizontal eye distances  
- Computes a ratio to determine eye closure  
- Increments counter upon each blink  

Blink data integrates with engagement score for deeper behavioral analysis.

---

## ⚙️ 6. Performance Tips

- Use a **GPU** for real-time performance.  
- Reduce input resolution or disable blink detection for faster FPS.  
- `mobilenetv2` offers faster inference than `resnet` models.

---

## 📊 7. Fine-Tuning (Optional)

You can fine-tune the model on another dataset:
```bash
python main.py --model resnet34 --dataset gaze360 --weight weights/resnet34.pt --epochs 5 --lr 1e-4
```

---

## 🧾 8. Example Outputs

- Live gaze visualization with bounding boxes  
- Engagement score overlay (0–100)  
- Blink counter display  

---

## 🧑‍💻 Author
**Mario Gamal Sobhy**  
AI Engineer | Computer Vision | Deep Learning  
[GitHub Repository](https://github.com/Mario-Gamal-Sobhy/SEMSOL)

---

## 🪪 License
This project is open-source under the MIT License.
