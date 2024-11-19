import torch

lin = torch.nn.Linear(4, 4)
x = torch.rand(4, 4)
print('Input(4X4):')
print(x)

print('\n\nWeight and Bias parameters:')
for param in lin.parameters():
    print(param)

y = lin(x)
print('\n\nOutput:')
print(y)
