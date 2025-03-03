import torch
import torch.nn as nn
from torch.optim import Adam
from data_loaders import create_data_loader

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

def test_training(db_path, table_name, epochs=10):
    # Create data loader
    train_loader = create_data_loader(db_path, table_name, batch_size=32)
    
    # Get input size from first batch
    sample_features, _ = next(iter(train_loader))
    input_size = sample_features.shape[1]
    
    # Initialize model and training components
    model = SimpleModel(input_size)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (features, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = (outputs >= 0.5).float()
            correct = (predictions == targets.float().unsqueeze(1)).sum().item()
            acc = correct / len(targets)
            print(f"Batch {batch_idx}, Accuracy: {acc:.4f}")
        # Print epoch statistics
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    return model

if __name__ == "__main__":
    # Example usage
    DB_PATH = "medinet.db"
    TABLE_NAME ="heart_failure_clinical_records_dataset"

    trained_model = test_training(DB_PATH, TABLE_NAME)
    print("Training completed successfully!")
