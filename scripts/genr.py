# 
# Slightly extended (added IV, OI to data featues)
# generates 300k random data sets for pricing an Option
# TODO
# - add BS or Heston price function to calculate option price
# - run some calcs to test model accuracy
#
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import time

# Step 1: Generate fake data as done previously
np.random.seed(42)
n = 10000
dates = pd.date_range(start='2020-01-01', periods=n, freq='D')

# Generate random data
asset_price = np.random.uniform(0.5, 750, n)
strike_price = np.random.uniform(0.5, 750, n)
risk_free_rate = np.random.uniform(0.05, 0.05, n)
volatility = np.random.uniform(0.1, 0.5, n)
time_to_maturity = np.random.uniform(0.01, 1, n)
dividend_yield = np.random.uniform(0.0, 0.00, n)
implied_volatility = np.random.uniform(0.1, 0.6, n)
volume = np.random.randint(1, 1000, n)
open_interest = np.random.randint(1, 5000, n)

# Create DataFrame
df = pd.DataFrame({
    'Asset_Price': asset_price,
    'Strike_Price': strike_price,
    'Risk_Free_Rate': risk_free_rate,
    'Volatility': volatility,
    'Time_to_Maturity': time_to_maturity,
    'Dividend_Yield': dividend_yield,
    'Implied_Volatility': implied_volatility,
    'Volume': volume,
    'Open_Interest': open_interest
})

# Step 2: Convert the DataFrame into PyTorch tensors
features = df.values.astype(np.float32)
target = np.random.uniform(50, 150, n).astype(np.float32)  # Fake target prices for demonstration

# Step 3: Define a custom PyTorch Dataset
class OptionPricingDataset(Dataset):
    def __init__(self, features, target):
        self.features = torch.tensor(features)
        self.target = torch.tensor(target)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

# Create the dataset and data loader
dataset = OptionPricingDataset(features, target)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 4: Define a simple neural network model
class OptionPricingModel(nn.Module):
    def __init__(self):
        super(OptionPricingModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize the model, loss function, and optimizer
model = OptionPricingModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Training loop
epochs = 128
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_features, batch_target in data_loader:
        optimizer.zero_grad()
        predictions = model(batch_features)
        loss = criterion(predictions.squeeze(), batch_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader)}")
    time.sleep(1)
torch.save(model.state_dict(), 'option_pricing_model_weights.pth')
print("Training complete!")
