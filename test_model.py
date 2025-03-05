import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import numpy as np
from data_loaders import create_train_val_test_loaders

class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def train_and_evaluate_model(db_path, table_name, epochs=10, val_size=0.15, test_size=0.15):
    """
    Train, validate and test a model using the SQLite dataset
    
    Args:
        db_path: Path to the SQLite database
        table_name: Name of the table to use
        epochs: Number of training epochs
        val_size: Proportion of data to use for validation
        test_size: Proportion of data to use for testing
        
    Returns:
        Trained model and performance metrics
    """
    # Create data loaders with train/val/test split
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        db_path, table_name, val_size=val_size, test_size=test_size, batch_size=32
    )
    
    # Get input size from first batch
    sample_features, _ = next(iter(train_loader))
    input_size = sample_features.shape[1]
    
    # Initialize model and training components
    model = SimpleModel(input_size)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # To track best model
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop with validation
    for epoch in tqdm(range(epochs), desc="Training"):
        # Training phase
        model.train()
        train_loss = 0
        train_acc = 0
        train_batches = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Calculate accuracy
            predictions = (outputs >= 0.5).float()
            correct = (predictions == targets.float().unsqueeze(1)).sum().item()
            train_acc += correct / len(targets)
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        avg_train_acc = train_acc / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_acc = 0
        val_batches = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                loss = criterion(outputs, targets.float().unsqueeze(1))
                val_loss += loss.item()
                
                # Calculate accuracy
                predictions = (outputs >= 0.5).float()
                correct = (predictions == targets.float().unsqueeze(1)).sum().item()
                val_acc += correct / len(targets)
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        avg_val_acc = val_acc / val_batches
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
    
    # Load best model for final evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    model.eval()
    test_loss = 0
    test_acc = 0
    test_batches = 0
    
    with torch.no_grad():
        for features, targets in test_loader:
            outputs = model(features)
            loss = criterion(outputs, targets.float().unsqueeze(1))
            test_loss += loss.item()
            
            # Calculate accuracy
            predictions = (outputs >= 0.5).float()
            correct = (predictions == targets.float().unsqueeze(1)).sum().item()
            test_acc += correct / len(targets)
            test_batches += 1
    
    avg_test_loss = test_loss / test_batches
    avg_test_acc = test_acc / test_batches
    
    print("\nFinal Test Results:")
    print(f"  Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.4f}")
    
    return model, {'test_loss': avg_test_loss, 'test_accuracy': avg_test_acc}

# For backward compatibility
def test_training(db_path, table_name, epochs=10):
    """Legacy function maintained for backward compatibility"""
    model, _ = train_and_evaluate_model(db_path, table_name, epochs=epochs)
    return model

if __name__ == "__main__":
    # Example usage
    DB_PATH = "medinet.db"
    TABLE_NAME = "heart_failure_clinical_records_dataset"

    trained_model, metrics = train_and_evaluate_model(DB_PATH, TABLE_NAME)
    print("Training completed successfully!")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
