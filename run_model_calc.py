import torch
import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, q=0, option_type="call") -> float:
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
   
    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price

# Define your model architecture (BlackScholes_ANN should match the architecture used during training)
class BlackScholes_ANN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(4, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 1),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

# Initialize the model
model = BlackScholes_ANN()

# Load the saved weights
path = "bs_model_weights.pth"
model.load_state_dict(torch.load(path, weights_only=True))

# Set the model to evaluation mode
model.eval()

inputs = [100.0, 0.5, 0.00, 0.2] # example: [stock price, time to expiry, dividends, volatility]
RATE = 0.05

# Adjust the shape according to your input dimensions.
sample_input = torch.tensor([inputs])


S        = inputs[0]      # stock price
K        = inputs[0] # strike
t        = inputs[1]      # tau
r        = RATE           # risk-free rate
vol      = inputs[3]      # vol
q        = inputs[2]      # dividends

for type in ["call","put"]:
    print(f"\n\nType: {type}")
    min = 1000
    for x in range(5,15):
        strike = K * float(x)/10.00 
        bs_result = black_scholes(S, strike, t, r, vol, q, type)
        #print(f"BS price: {bs_result:.12f}")

        with torch.no_grad():
            prediction = model(sample_input)
        #print(f"ANN price: {prediction.item():.12f}")
        gap = bs_result-prediction.item()
        #print(f"Gap: {gap:.12f} for {x}")
        if abs(gap) < min:
            min = abs(gap)
            factor = x
    print(f"found min {type} gap {min:.4f} @ {100.00*factor/10.00:.2f}%")
