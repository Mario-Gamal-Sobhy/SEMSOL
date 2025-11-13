from src.pipeline.inference_pipeline import InferencePipeline

model = InferencePipeline()
text = model.predict("5639-40744-0032.wav")
print(text)