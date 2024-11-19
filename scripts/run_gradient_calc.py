import torch
# set up simple graph relating x, y and z
x = torch.tensor(3.5, requires_grad=True)
x2 = torch.tensor(4.5, requires_grad=True)
x3 = torch.tensor(5.5, requires_grad=True)
y = x*x
y2 = x2*x2
z = 2*y + 3
z2 = 2*y2 + 3
print("x: ", x)
print("x2: ", x2)
print("y = x*x: ", y)
print("z= 2*y + 3: ", z)
# work out gradients
z.backward()
z2.backward()
print("Working out gradients dz/dx")
# what is gradient at x = 3.5
print("Gradient at x = 3.5: ", x.grad)
print("Gradient at x2 = 4.5: ", x2.grad)
