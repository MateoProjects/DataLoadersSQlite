import torch
from torch.utils.data import Dataset, DataLoader
import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data_from_sqlite(db_path, table_name):
    """
    Load data from a SQLite database table
    Returns metadata and the dataframe with the data
    """
    with sqlite3.connect(db_path) as conn:
        # Load metadata
        metadata_df = pd.read_sql(
            f"SELECT * FROM metadata WHERE dataset_name = '{table_name}'", 
            conn
        )
        target_column = metadata_df['target_column'].iloc[0]
        
        # Load all data at once
        data_df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    
    return metadata_df, data_df, target_column

class SQLiteDataset(Dataset):
    def __init__(self, features, targets):
        """
        Initialize a dataset with pre-loaded features and targets
        
        Args:
            features: Feature tensor (X)
            targets: Target tensor (y)
        """
        self.X = features
        self.y = targets
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_dataset(data_df, target_column):
    """
    Prepare dataset from a pandas DataFrame
    
    Args:
        data_df: DataFrame with all data
        target_column: Name of the target column
        
    Returns:
        X_tensor: Feature tensor
        y_tensor: Target tensor
    """
    X_tensor = torch.FloatTensor(
        data_df.drop(columns=[target_column]).values
    )
    y_tensor = torch.tensor(data_df[target_column].values)
    
    return X_tensor, y_tensor


def create_train_val_loaders(db_path, table_name, val_size=0.2, 
                           batch_size=32, random_state=42):
    """
    Create only train and validation dataloaders from SQLite database
    
    Args:
        db_path: Path to SQLite DB
        table_name: Name of the table to load
        val_size: Size of validation set (proportion)
        batch_size: Batch size for dataloaders
        random_state: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader
    """
    # Load data from SQLite
    metadata_df, data_df, target_column = load_data_from_sqlite(db_path, table_name)
    
    # Split data into train and validation sets only
    train_df, val_df = train_test_split(
        data_df, test_size=val_size, random_state=random_state
    )
    
    # Prepare tensors
    X_train, y_train = prepare_dataset(train_df, target_column)
    X_val, y_val = prepare_dataset(val_df, target_column)
    
    # Create datasets
    train_dataset = SQLiteDataset(X_train, y_train)
    val_dataset = SQLiteDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
