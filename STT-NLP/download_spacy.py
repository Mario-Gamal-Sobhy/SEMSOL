
import spacy
import subprocess

def download_spacy_model(model_name="en_core_web_sm"):
    """
    Downloads and installs a spaCy model if it's not already installed.
    """
    try:
        spacy.load(model_name)
        print(f"spaCy model '{model_name}' already installed.")
    except OSError:
        print(f"spaCy model '{model_name}' not found. Downloading and installing...")
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        print(f"spaCy model '{model_name}' downloaded and installed successfully.")

if __name__ == '__main__':
    download_spacy_model()
