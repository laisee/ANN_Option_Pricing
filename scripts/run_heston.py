import torch
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)
print(f"Adding {parent_dir} to sys.path")

from models.heston.heston import Heston_ANN, Heston as heston

print("loading Heston ANN model")
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

S = 100.00
L = 50
N = 1500

inputs = [S, 0.5, 0.00, 0.2] # example: [stock price, time to expiry, dividends, volatility]
RATE = 0.05
print(f"Calculating price using Heston AMM with inputs {inputs}")

# Adjust the shape according to your input dimensions.
sample_input = torch.tensor([inputs])


# Latin Hypercube Sampling for the Heston model
M_H = [0.6,1.4] # moneyness = S0/K
TAU_H = [0.1,1.4]
R_H = [0.0,0.1]
RHO = [-0.95,0.0]
MRS = [0.0,2.0]
V_BAR = [0.0,0.5]
VOLVOL = [0.0,0.5]
SIGMA_H = [0.05,0.5]


S        = inputs[0]      # stock price
K        = inputs[0]      # strike
t        = inputs[1]      # tau
r        = RATE           # risk-free rate
vol      = inputs[3]      # vol
q        = inputs[2]      # dividends

for type in ["call"]:
    min = 1000
    for x in range(5,15):
        strike = K * float(x)/10.00 
        heston_result = heston(S,strike,t, MRS[0], r, vol, V_BAR[0], RHO, SIGMA_H, L, N)
        print(f"Heston price: {heston_result:.12f}")

        with torch.no_grad():
            prediction = model(sample_input)
        #print(f"ANN(Heston) price: {prediction.item():.12f}")
        gap = heston_result-prediction.item()
        #print(f"Gap: {gap:.12f} for {x}")
        if abs(gap) < min:
            min = abs(gap)
            factor = x
    print(f"found min {type} gap {min:.4f} @ {100.00*factor/10.00:.2f}%")
