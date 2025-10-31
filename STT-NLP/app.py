import streamlit as st
from src.pipeline.inference_pipeline import InferencePipeline
import io

st.set_page_config(page_title="STT-NLP Transcription", page_icon="🎤", layout="centered")
st.title("Speech-to-Text (STT-NLP)")

@st.cache_resource
def load_pipeline():
    return InferencePipeline()

pipe = load_pipeline()

st.subheader("Upload an audio file (.wav)")
uploaded = st.file_uploader("Choose a WAV file", type=["wav"]) 

if uploaded is not None:
    st.audio(uploaded)
    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            buf = io.BytesIO(uploaded.read())
            text = pipe.predict(buf)
        st.success("Done")
        st.write("Transcription:")
        st.text_area("", value=text, height=200)

st.caption("Model and preprocessor are loaded from artifacts as configured in config.yaml.")