import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

dataset = np.loadtxt('scripts/pima_data.csv', delimiter=',')
X = dataset[:,0:8]
Y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)

model = nn.Sequential(
    nn.Linear(8, 122),
    nn.ReLU(),
    nn.Linear(122, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 400
batch_size = 10
 
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = Y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

with torch.no_grad():
    y_pred = model(X)
 
accuracy = 100.00*(y_pred.round() == Y).float().mean()
print(f"Accuracy {accuracy:.4f}%")

predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))
