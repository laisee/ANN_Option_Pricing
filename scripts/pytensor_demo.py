import torch
# set up simple graph relating x, y and z
a = torch.tensor(2.0, requires_grad=True)
print(f"added tensor 'a' -> {a}")
b = torch.tensor(1.0, requires_grad=True)
print(f"added tensor 'b' -> {b}")
x = 2*a + 3*b
y = 5*a*a + 3*b*b*b
z = 2*x + 3*y
print(f"added eqn 'x' -> {x} = 2*a + 3*b")
print(f"added eqn 'y' -> {y} = 5*a*a + 3*b*b*b*b")
print(f"added eqn 'z' -> {z} = 2*x + 3*y")
# work out gradients
print("\nCalculating 'z' gradients ...")
z.backward()
print("\nWorking out gradient dz/da for a = 2.00")
print(f"\nGradient at a=2.0: -> {a.grad}")
