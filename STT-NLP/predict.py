from src.pipeline.inference_pipeline import InferencePipeline

model = InferencePipeline()
text = model.predict("5639-40744-0001.wav")
print(text)