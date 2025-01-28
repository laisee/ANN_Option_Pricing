from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

class BlackScholes_ANN(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_relu_stack = nn.Sequential(

        nn.Linear(4,400),
        nn.ReLU(),

        nn.Linear(400,400),
        nn.ReLU(),

        nn.Linear(400,400),
        nn.ReLU(),

        nn.Linear(400,400),
        nn.ReLU(),

        nn.Linear(400,1),
    )

  def forward(self,x):
    target = self.linear_relu_stack(x)
    return target

  def weights_init(self): # initializing parameters using glorot uniform unitialization
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)

WEIGHTS_FILE = "models/bs_model_weights.pth"
model = BlackScholes_ANN()
model.load_state_dict(torch.load(WEIGHTS_FILE,weights_only=True))
model.eval()

print(model)

S = 100
K = np.arange(80,140,1)
tau = 1
r = 0.05
sigma = 0.25

inputs = np.zeros((len(K),4))

for i in range(len(K)):
  inputs[i,:] = [S/K[i],tau,r,sigma]
inputs = torch.tensor(inputs).type(torch.float)

output = model(inputs).detach().numpy()

# Print the predicted value
#print(output.detach().numpy())

plt.plot(K,output,'b',linestyle='dashed',label="BS-ANN")
plt.xlabel("Strike K")
plt.ylabel("Price")
plt.legend()
plt.show()
