import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

class CorrelationNet(nn.Module):
    def __init__(self, input_size):
        super(CorrelationNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)  # Output layer (for single output)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():

    # Load data from CSV
    file_path = 'ann_price_calc/correlation/correlation_data.csv'
    correlation_data = pd.read_csv(file_path)
    
    # Encode categorical columns
    label_encoders = {col: LabelEncoder().fit(correlation_data[col]) for col in ['Symbol', 'MostCorrelatedWith', 'LeastCorrelatedWith']}
    for col, encoder in label_encoders.items():
        correlation_data[col] = encoder.transform(correlation_data[col])
    
    # Standardize numerical columns
    scaler = StandardScaler()
    numerical_features = ['AvgCorrelation', 'MaxCorrelation', 'MinCorrelation']
    correlation_data[numerical_features] = scaler.fit_transform(correlation_data[numerical_features])
    
    # Convert processed DataFrame to a PyTorch tensor
    features = torch.tensor(correlation_data.values, dtype=torch.float32)
    
    # Suppose target is a regression target for each cryptocurrency (example only)
    # Generate synthetic target data for demonstration (e.g., some random values)
    # Replace this with your actual target data
    target = torch.randn(features.shape[0], 1)  # shape [number of samples, 1]
    
    # Create a DataLoader for batch processing
    dataset = TensorDataset(features, target)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize the model, loss function, and optimizer
    model = CorrelationNet(input_size=features.shape[1])
    criterion = nn.MSELoss()  # Use MSE for regression; use CrossEntropyLoss for classification
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 800  # Adjust as necessary
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0
        for batch_features, batch_target in data_loader:
            # Forward pass
            predictions = model(batch_features)
            loss = criterion(predictions, batch_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Print average loss for each epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(data_loader)}")
    
    torch.save(model.state_dict(), 'ann_price_calc/correlation/correlation_net_weights.pth')
    
    model = CorrelationNet(input_size=features.shape[1])  # Adjust input size if needed
    model.load_state_dict(torch.load('ann_price_calc/correlation/correlation_net_weights.pth', weights_only=True))
    model.eval()
    print(model)
    
    print("Making predictions using trained model")
    with torch.no_grad():  # Disable gradient calculation for evaluation
        predictions = model(features).numpy()  # Convert predictions to numpy
        true_values = target.numpy()  # Convert true target values to numpy
    
    print("Calculating prediction accuracy")
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

if __name__ == "__main__":
    print("Running Correaltion ANN Model ...")
    main()
