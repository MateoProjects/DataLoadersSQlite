import torch
from torch.utils.data import Dataset, DataLoader
import sqlite3
import numpy as np
import pandas as pd

class SQLiteDataset(Dataset):
    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.table_name = table_name
        
        # Read metadata
        with sqlite3.connect(db_path) as conn:
            metadata_df = pd.read_sql(
                f"SELECT * FROM metadata WHERE dataset_name = '{table_name}'", 
                conn
            )
            self.n_rows = metadata_df['n_rows'].iloc[0]
            self.n_columns = metadata_df['n_columns'].iloc[0]
            self.target_column = metadata_df['target_column'].iloc[0]
            
            # Load all data at once
            self.data_df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            
            # Separate features and target
            self.X = torch.FloatTensor(
                self.data_df.drop(columns=[self.target_column]).values
            )
            self.y = torch.tensor(self.data_df[self.target_column].values)
    
    def __len__(self):
        return self.n_rows
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Example usage:
def create_data_loader(db_path, table_name, batch_size=32, shuffle=True):
    dataset = SQLiteDataset(db_path, table_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
