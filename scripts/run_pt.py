import torch
import torch.nn as nn

# Define your model class
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(5, 10)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# Initialize model and load state dict
model = MyModel()
model.load_state_dict(torch.load("BS_data.pt"))
model.eval()

# Prepare input data
input_tensor = torch.randn(1000000, 5)

# Perform inference
with torch.no_grad():
    output = model(input_tensor)

print(output)

