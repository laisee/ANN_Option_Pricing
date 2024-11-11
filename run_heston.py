import numpy as np
import torch
from models.heston import Heston_ANN as Hest_ANN
from models.heston import Heston as Heston
from models.heston import Heston_2 as Heston2
from smt.sampling_methods import LHS

print("running heston model ...")

S = 100
L = 50
N = 1500

# Latin Hypercube Sampling for the Heston model
M_H = [0.6,1.4] # moneyness = S0/K
TAU_H = [0.1,1.4]
R_H = [0.0,0.1]
RHO = [-0.95,0.0]
MRS = [0.0,2.0]
V_BAR = [0.0,0.5]
VOLVOL = [0.0,0.5]
SIGMA_H = [0.05,0.5]
print("Initialized heston model params ...")

H_param_space = np.array([M_H,TAU_H,MRS,R_H,VOLVOL,V_BAR,RHO,SIGMA_H])
sampling_H = LHS(xlimits=H_param_space)
num_H = 10**6
x_H = sampling_H(num_H)

labeled_H = np.zeros((num_H,9))
for i in range(num_H):
    print(f"running iteration {i} for calculating heston")
    m,tau,mrs,r,volvol,v_bar,rho,sigma = x_H[i,0],x_H[i,1],x_H[i,2],x_H[i,3],x_H[i,4],x_H[i,5],x_H[i,6],x_H[i,7]
    price = Heston2(S,S/m,tau,mrs,r,volvol,v_bar,rho,sigma,L,N)
    print(f"generated price {price} ... ")
    labeled_H[i,:] = [m,tau,mrs,r,volvol,v_bar,rho,sigma,price]

print(f"completed price calc ... ")
heston_data = torch.tensor(labeled_H).type(torch.float)
print(f"saving heston model ... ")
torch.save(heston_data, 'models/heston_data.pt')
print(f"saved heston model ... ")
print("completed heston model calcs")
