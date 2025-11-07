from src.pipeline.inference_pipeline import InferencePipeline


if __name__=='__main__':
    model = InferencePipeline()
    text = model.predict("test_audio.wav")
    print(text)