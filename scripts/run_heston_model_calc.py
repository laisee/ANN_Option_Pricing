import torch
import numpy as np
import os
from scipy.stats import norm
from models.heston import Heston_ANN, Heston as heston

print("loading Heston AMM model")
# Initialize the model
model = Heston_ANN()

# Load the saved weights
print("loading Heston AMM eights if they exist")
path = "models/heston_model_weights.pth"
if os.path.exists(path):
    model.load_state_dict(torch.load(path, weights_only=True))
else:
    print(f"No weights fule found using '{path}' ")

# Set the model to evaluation mode
model.eval()

inputs = [100.0, 0.5, 0.00, 0.2] # example: [stock price, time to expiry, dividends, volatility]
RATE = 0.05
print(f"Calculating price using Heston AMM with inputs {inputs}")

# Adjust the shape according to your input dimensions.
sample_input = torch.tensor([inputs])


S        = inputs[0]      # stock price
K        = inputs[0]      # strike
t        = inputs[1]      # tau
r        = RATE           # risk-free rate
vol      = inputs[3]      # vol
q        = inputs[2]      # dividends

for type in ["call","put"]:
    print(f"\n\nType: {type}")
    min = 1000
    for x in range(5,15):
        strike = K * float(x)/10.00 
        heston_result = heston(S, strike, t, r, vol, q, type)
        print(f"BS price: {heston_result:.12f}")

        with torch.no_grad():
            prediction = model(sample_input)
        #print(f"ANN price: {prediction.item():.12f}")
        gap = heston_result-prediction.item()
        #print(f"Gap: {gap:.12f} for {x}")
        if abs(gap) < min:
            min = abs(gap)
            factor = x
    print(f"found min {type} gap {min:.4f} @ {100.00*factor/10.00:.2f}%")