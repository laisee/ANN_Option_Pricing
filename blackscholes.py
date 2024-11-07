from torch import nn
from scipy.stats import norm
import numpy as np

def bs(S: float, K: float, tau: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Generates price of a European option (call or put) using the Black-Scholes formula.
    
    Parameters:
    - S: float, Spot price of the underlying asset
    - K: float, Strike price of the option
    - tau: float, Time to maturity (in years)
    - r: float, Risk-free interest rate
    - sigma: float, Volatility of the underlying asset
    - option_type: str, "call" for call option, "put" for put option (default is "call")
    
    Returns:
    - V: float, Option price
    """
    
    d_1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d_2 = d_1 - sigma * np.sqrt(tau)
    
    if option_type == "call":
        V = norm.cdf(d_1) * S - norm.cdf(d_2) * K * np.exp(-r * tau)
    elif option_type == "put":
        V = norm.cdf(-d_2) * K * np.exp(-r * tau) - norm.cdf(-d_1) * S
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return V

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
