
import pandas as pd

class DataIngestion:
    def __init__(self, data_path):
        self.data_path = data_path

    def get_data(self):
        """
        Reads the data from the given path and returns a pandas DataFrame.
        """
        return pd.read_csv(self.data_path, encoding='latin1')

