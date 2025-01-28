import os
import sys
import torch

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)
from models.tiny_model import TinyModel

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

print("saving weights for Tiny Model to 'models/tinymodel.pth'")
torch.save(tinymodel.state_dict(), "tinymodel.pth")
print("completed executing Tiny Model")
