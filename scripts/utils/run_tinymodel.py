import os
import sys
import torch

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Use absolute import instead of relative import
from models.tiny.tiny_model import TinyModel

# Create an instance of the TinyModel
tinymodel = TinyModel()

print(f"Tiny Model: {tinymodel}")

print('\n\nJust one layer:')
print(tinymodel.linear2)

print('\n\nModel params:')
for param in tinymodel.parameters():
    print(param)

print('\n\nLayer params:')
for param in tinymodel.linear2.parameters():
    print(param)

# Create models directory if it doesn't exist
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# Save to the correct path
model_path = os.path.join(models_dir, "tinymodel.pth")
print(f"saving weights for Tiny Model to '{model_path}'")
torch.save(tinymodel.state_dict(), model_path)
print("completed executing Tiny Model")
