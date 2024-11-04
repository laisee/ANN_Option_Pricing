import torch
from scripts.tinymodel import TinyModel
import warnings

# Suppress FutureWarning from torch
warnings.filterwarnings("ignore", category=FutureWarning)

tm = TinyModel()
tm.load_state_dict(torch.load("tinymodel.pth"))
tm.eval()
print(tm)

test_input = torch.randn(1, 100, 100 )

# Perform a forward pass
with torch.no_grad():  # Disables gradient calculation as it's not needed for inference
    output = tm(test_input)

print("Model output:", output)
