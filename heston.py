from torch import nn
import numpy as np

def charFuncHestonFO(u,tau,mrs,r,volvol,v_bar,rho,sigma):
    '''
    The characteristic function of the Heston log-asset price evaluated at u
    '''
    d = np.sqrt(np.power(mrs - 1j*rho*volvol*u, 2) + np.power(volvol,2) * (np.power(u,2) + u*1j))
    g = (mrs - 1j*rho*volvol*u - d)/(mrs - 1j*rho*volvol*u + d)
    C = (mrs*v_bar/np.power(volvol,2)) * ((mrs - 1j*rho*volvol*u - d)*tau - 2*np.log((1 - g * np.exp(-d * tau))/(1-g)))
    D = 1j*r*u*tau + (sigma/np.power(volvol,2)) * ((1 - np.exp(-d*tau))/(1 - g*np.exp(-d*tau))) * (mrs - 1j*rho*volvol*u - d)
    phi = np.exp(D) * np.exp(C)
    return phi

class Heston_ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(

          nn.Linear(8,400),
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

def Heston(S,K,tau,mrs,r,volvol,v_bar,rho,sigma,L,N):
  '''
  Calculates the price of a european call option with a Heston asset and using the COS method
  Initial asset price S, time to maturity tau, strike K with the rest of the Heston parameters
  '''
  k = np.arange(N)
  x = np.log(S/K)
  a,b = truncation(L,tau,mrs,r,volvol,v_bar,rho,sigma)
  u = k*np.pi/(b-a)

  phi_heston = charFuncHestonFO(u,tau,mrs,r,volvol,v_bar,rho,sigma)
  ExpTerm = np.exp(1j*k*np.pi*(x-a)/(b-a))
  Fk = np.real(phi_heston*ExpTerm)
  Fk[0] = 0.5*Fk[0]
  UkCall = 2/(b-a)*(cosSerExp(a,b,0,b,k) - cosSer1(a,b,0,b,k))

  return K*np.exp(-r*tau)*np.sum(np.multiply(Fk,UkCall))

def Heston_2(S,K,tau,mrs,r,volvol,v_bar,rho,sigma,L,N):
    '''
    Calculates the price of a european call option with a Heston asset and using the COS method but with put-call parity.
    In fact, COS method doesn't give good results for deep OTM calls, so we'll use this function to generate our data
    '''
    k = np.arange(N)
    x = np.log(S/K)
    a,b = truncation(L,tau,mrs,r,volvol,v_bar,rho,sigma)
    u = k*np.pi/(b-a)

    phi_heston = charFuncHestonFO(u,tau,mrs,r,volvol,v_bar,rho,sigma)
    ExpTerm = np.exp(1j*k*np.pi*(x-a)/(b-a))
    Fk = np.real(phi_heston*ExpTerm)
    Fk[0] = 0.5*Fk[0]
    UkPut  = 2/(b-a)*(cosSer1(a,b,a,0,k) - cosSerExp(a,b,a,0,k))
    V_Put = K * np.sum(Fk*UkPut)*np.exp(-r*tau)
    V_Call = V_Put + S - K*np.exp(-r*tau)
    return V_Call

def heston_implied_vol_(S,K,tau,mrs,r,volvol,v_bar,rho,sigma,L,N,a,b):
    '''
    Finds the B-S implied volatility using Heston otpions price as the "market price" and using Brent's root finding method from scipy.optimize
    '''
    g = lambda x : bs(S,K,tau,r,x)*K - Heston_2(S,K,tau,mrs,r,volvol,v_bar,rho,sigma,L,N)
    root = optimize.brentq(g, a, b)
    return root

def heston_implied_vol(S,K,tau,r,Heston_price):
    '''
    Finds the B-S implied volatility using Heston otpions prices that are already calculated as the "market price" and using Brent's root finding method from scipy.optimize
    '''
    a = -3
    b = 5
    g = lambda x : bs(S,K,tau,r,x)*K - Heston_price
    root = optimize.brentq(g, a, b)
    return root