# STT-NLP Weights

To use the STT model without training:

1) Host the trained files somewhere accessible:
- artifacts/model_trainer/model.pt
- artifacts/data_transformation/preprocessor.pkl

2) Download into this repo:
```
export STT_MODEL_URL=https://your.host/path/model.pt
export STT_PREPROC_URL=https://your.host/path/preprocessor.pkl
python weights/download_weights.py
```

3) Run inference:
```
streamlit run app.py
# or
python -c "from src.pipeline.inference_pipeline import InferencePipeline;print(InferencePipeline().predict('path.wav'))"
```
