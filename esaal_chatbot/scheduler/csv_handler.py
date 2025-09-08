import pandas as pd

class CSVHandler:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def read_schedule(self):
        return pd.read_csv(self.csv_path)

    def write_schedule(self, df):
        df.to_csv(self.csv_path, index=False)
