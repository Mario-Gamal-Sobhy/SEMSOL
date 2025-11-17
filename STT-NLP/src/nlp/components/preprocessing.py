
import spacy
import re
import pandas as pd

class TextPreprocessor:
    def __init__(self, model_name="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"spaCy model '{model_name}' not found. Please run 'python -m spacy download {model_name}' to install it.")
            self.nlp = None

    def preprocess_text(self, text):
        """
        Cleans and preprocesses a single text string.
        """
        if not self.nlp:
            raise RuntimeError("spaCy model not loaded. Please install it first.")
            
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
        text = text.lower()
        text = text.strip()
        
        doc = self.nlp(text)
        
        # Lemmatize, remove stop words and punctuation
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        return " ".join(tokens)

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str, sentiment_column: str, id_column: str = None):
        """
        Applies preprocessing to a DataFrame.
        """
        if id_column and id_column in df.columns:
            df = df.drop(columns=[id_column])
            
        df[text_column] = df[text_column].fillna('')
        df[text_column] = df[text_column].apply(self.preprocess_text)
        
        # Drop rows where text is empty after preprocessing
        df = df[df[text_column] != '']
        
        return df[[text_column, sentiment_column]]

