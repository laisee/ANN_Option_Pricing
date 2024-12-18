from tqdm import tqdm
import numpy as np
from scipy.stats import norm
from scipy import optimize
import smt
from smt.sampling_methods import LHS
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import plotly.express as px

def train_loop(dataloader, model, loss_fn, optimizer):
  for batch, (X,y) in enumerate(dataloader):
    # forward compute
    pred = model(X)
    loss = loss_fn(pred,y.unsqueeze(1))

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def BS(S,K,tau,r,sigma):
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

def truncation(L,tau,mrs,r,volvol,v_bar,rho,sigma):
  '''
  Finds the integration range for the COS method
  L : truncation parameter
  '''
  c1 = r* tau + (1 - np.exp(-mrs * tau)) * (v_bar - sigma)/(2 * mrs) - v_bar * tau / 2 # this is the first order cumulant of the characterisctic function of the log-asset price

  c2 = 1/(8 * np.power(mrs,3)) * (volvol * tau * mrs * np.exp(-mrs * tau) \
      * (sigma - v_bar) * (8 * mrs * rho - 4 * volvol) \
      + mrs * rho * volvol * (1 - np.exp(-mrs * tau)) * (16 * v_bar - 8 * sigma) \
      + 2 * v_bar * mrs * tau * (-4 * mrs * rho * volvol + np.power(volvol,2) + 4 * np.power(mrs,2)) \
      + np.power(volvol,2) * ((v_bar - 2 * sigma) * np.exp(-2 * mrs * tau) \
      + v_bar * (6 * np.exp(-mrs * tau) - 7) + 2 * sigma) \
      + 8 * np.power(mrs,2) * (sigma - v_bar) * (1 - np.exp(-mrs * tau))) # this is the second order cumulant of the characterisctic function of the log-asset price

  a = c1 - L * np.sqrt(np.abs(c2))
  b = c1 + L * np.sqrt(np.abs(c2))

  return a, b

def cosSerExp(a,b,c,d,k):
  '''
  The cosine series coefficients of g(y)=exp(y) on [c,d] included in [a,b]
  k : positive integer

  '''
  bma = b-a
  uu  = k * np.pi/bma
  chi =  (1/(1 + np.power(uu,2)))*(np.cos(uu*(d-a))*np.exp(d) - np.cos(uu*(c-a))*np.exp(c) + uu*np.sin(uu*(d-a))*np.exp(d) - uu*np.sin(uu*(c-a))*np.exp(c))

  return chi

def cosSer1(a,b,c,d,k):
  '''
  The cosine series coefficients of g(y)=1 on [c,d] included in [a,b]
  k : positive integer

  '''
  bma    = b-a
  uu     = k * np.pi/bma
  uu[0]  = 1
  psi    = (1/uu)*(np.sin(uu*(d-a)) - np.sin(uu*(c-a)))
  psi[0] = d-c
  return psi

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

  V = K*np.exp(-r*tau)*np.sum(np.multiply(Fk,UkCall))

  return V

def implied_vol_(S,K,tau,mrs,r,volvol,v_bar,rho,sigma,L,N,a,b):
  '''
  Finds the B-S implied volatility using Heston otpions price as the "market price" and using Brent's root finding method from scipy.optimize
  '''
  g = lambda x : BS(S,K,tau,r,x)*K - Heston_2(S,K,tau,mrs,r,volvol,v_bar,rho,sigma,L,N)
  root = optimize.brentq(g, a, b)
  return root

def implied_vol(S,K,tau,r,Heston_price):
  '''
  Finds the B-S implied volatility using Heston otpions prices that are already calculated as the "market price" and using Brent's root finding method from scipy.optimize
  '''
  a = -3
  b = 5
  g = lambda x : BS(S,K,tau,r,x)*K - Heston_price
  root = optimize.brentq(g, a, b)
  return root

# Epoch Count
EPOCHS = 50
S = 100
L = EPOCHS
N = 1500

# Latin Hypercube Sampling for the B-S model
M_BS = [0.4,1.6] # moneyness = S0/K
TAU_BS = [0.2,1.1]
R_BS = [0.02,0.1]
SIGMA_BS = [0.01,1.5]

BS_param_space = np.array([M_BS,TAU_BS,R_BS,SIGMA_BS])

sampling_BS = LHS(xlimits=BS_param_space)
num_BS = 10**6 # we will generate 1 million labeled data points
x_BS = sampling_BS(num_BS)

labeled_BS = np.zeros((num_BS,5))
for i in range(num_BS):
  m,tau,r,sigma = x_BS[i,0],x_BS[i,1],x_BS[i,2],x_BS[i,3]
  price = BS(S,S/m,tau,r,sigma)
  labeled_BS[i,:] = [m,tau,r,sigma,price]

BS_data = torch.tensor(labeled_BS).type(torch.float)


BS_ANN = BlackScholes_ANN()
BS_ANN.weights_init()
loss_function = nn.MSELoss()
batch_size = 128
epochs = EPOCHS
optimizer = torch.optim.Adam(BS_ANN.parameters(), lr=10e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # after 10 epochs the new lr is lr*gamma

train_size = int(0.8*num_BS)
X = BS_data[:,:-1] # inputs of the model
Y = BS_data[:,-1] # the price
Xtrain, Xtest = X[:train_size,:], X[train_size:,:]
Ytrain, Ytest = Y[:train_size], Y[train_size:]

train_set = TensorDataset(Xtrain,Ytrain)
train_dataloader = DataLoader(train_set, batch_size=batch_size)

training_loss = []
test_loss = []
for epoch in tqdm(range(epochs)):
  print(f"executing epoch {epoch}")
  running_training_loss = train_loop(train_dataloader, BS_ANN, loss_function, optimizer)
  running_test_loss = loss_function(BS_ANN(Xtest),Ytest.unsqueeze(1))
  running_training_loss = loss_function(BS_ANN(Xtrain),Ytrain.unsqueeze(1))
  test_loss.append(running_test_loss.detach().numpy())
  training_loss.append(running_training_loss.detach().numpy())
  scheduler.step()

torch.save(BS_ANN.state_dict(), 'bs_model_weights.pth')

plt.figure(figsize=(9,3))
plt.plot(np.arange(1,EPOCHS + 1,1),np.log(np.array(training_loss)),'r',label="Training loss")
plt.plot(np.arange(1,EPOCHS + 1,1),np.log(np.array(test_loss)),'b',linestyle='dashed',label="Test loss")
plt.xlabel("Epochs")
plt.ylabel("Log MSE loss")
plt.legend()
plt.show()
# plt.savefig("MSE_loss_BS_1mil_50_epochs.png",dpi=1200)

S = 100
K = np.arange(65,190,1)
tau = 1
r = 0.05
sigma = 0.5
inputs = np.zeros((len(K),4))

for i in range(len(K)):
  inputs[i,:] = [S/K[i],tau,r,sigma]

inputs = torch.tensor(inputs).type(torch.float)

real = BS(S,K,tau,r,sigma)
model = BS_ANN(inputs).detach().numpy()
plt.plot(K,real,'r',label="Real BS function")
plt.plot(K,model,'b',linestyle='dashed',label="BS-ANN")
plt.xlabel("Strike K")
plt.ylabel("Scaled price V/K")
plt.legend()
plt.show()
# plt.savefig("predictions_BS_1mil_50_epochs.png",dpi=1200)
