import os
import sys
import torch
import warnings

# Suppress FutureWarning from torch
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from models.tiny.tiny_model import TinyModel

tm = TinyModel()
tm.load_state_dict(torch.load("models/tinymodel.pth"))
tm.eval()
print(tm)

test_input = torch.randn(1, 100, 100 )

# Perform a forward pass
with torch.no_grad():  # Disables gradient calculation as it's not needed for inference
    output = tm(test_input)

print("Model output:", output)
