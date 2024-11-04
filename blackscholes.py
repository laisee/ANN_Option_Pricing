from torch import nn
from scipy.stats import norm
import numpy as np

def bs(S,K,tau,r,sigma):
  '''
  Gives the scaled price of a european option as given by the closed B-S formula
  '''
  d_1 = np.divide((np.log(np.divide(S,K)) + (r+0.5*np.multiply(np.power(sigma,2),tau))), (np.multiply(sigma,np.sqrt(tau))))
  d_2 = d_1 - np.multiply(sigma,np.sqrt(tau))

  V = np.multiply(norm.cdf(x=d_1),S) - np.multiply(np.multiply(norm.cdf(x=d_2),K),np.exp(-np.multiply(r,tau)))
  return V/K

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
  
def train_loop(dataloader, model, loss_fn, optimizer):
    for batch, (X,y) in enumerate(dataloader):
        # forward compute
        pred = model(X)
        loss = loss_fn(pred,y.unsqueeze(1))

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
