import torch
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)
print(f"Adding {parent_dir} to sys.path")

from models.heston.heston import Heston_ANN, Heston as heston

print("loading Heston ANN model")
# Initialize the model
model = Heston_ANN()

# Load the saved weights
print("loading Heston AMM weights if they exist")
path = "models/heston_model_weights.pth"
if os.path.exists(path):
    model.load_state_dict(torch.load(path, weights_only=True))
else:
    print(f"No weights file found using '{path}' ")