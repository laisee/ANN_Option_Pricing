from tqdm import tqdm
import numpy as np
from scipy.stats import norm

from heston import Heston_ANN, Heston_2, heston_implied_vol_, heston_implied_vol
from blackscholes import bs, BlackScholes_ANN, train_loop
from utils import truncation

from smt.sampling_methods import LHS
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import plotly.express as px

#
# Model settings
#
EPOCHS = 25

def price(asset_price,strike,expiry,rate,vol, call_type="call"):
    return bs(asset_price,strike,expiry,rate,vol)
#
# Run BS ANN
#
def run_bs():

    S = 100
    L = 50
    N = 1500

    # Latin Hypercube Sampling for the B-S model
    M_BS = [0.4,1.6] # moneyness = S0/K
    TAU_BS = [0.2,1.1]
    R_BS = [0.02,0.1]
    SIGMA_BS = [0.01,1.0]

    BS_param_space = np.array([M_BS,TAU_BS,R_BS,SIGMA_BS])

    sampling_BS = LHS(xlimits=BS_param_space)
    num_BS = 10**6 # generate 1 million labeled data points
    x_BS = sampling_BS(num_BS)

    labeled_BS = np.zeros((num_BS,5))
    for i in range(num_BS):
        m,tau,r,sigma = x_BS[i,0],x_BS[i,1],x_BS[i,2],x_BS[i,3]
        price = bs(S,S/m,tau,r,sigma)
        labeled_BS[i,:] = [m,tau,r,sigma,price]

    BS_data = torch.tensor(labeled_BS).type(torch.float)

    torch.save(BS_data, 'bs_data.pt')

    print("setting up BS ANN")
    BS_ANN = BlackScholes_ANN()
    BS_ANN.weights_init()
    loss_function = nn.MSELoss()
    batch_size = 1000
    epochs = EPOCHS
    optimizer = torch.optim.Adam(BS_ANN.parameters(), lr=10e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # after 10 epochs the new lr is lr*gamma

    train_size = int(0.9*num_BS)
    X = BS_data[:,:-1] # inputs of the model
    Y = BS_data[:,-1] # the price
    Xtrain, Xtest = X[:train_size,:], X[train_size:,:]
    Ytrain, Ytest = Y[:train_size], Y[train_size:]

    train_set = TensorDataset(Xtrain,Ytrain)
    train_dataloader = DataLoader(train_set, batch_size=batch_size)

    training_loss = []
    test_loss = []
    print("running training for BS ANN")
    for epoch in tqdm(range(epochs)):
        running_training_loss = train_loop(train_dataloader, BS_ANN, loss_function, optimizer)
        running_test_loss = loss_function(BS_ANN(Xtest),Ytest.unsqueeze(1))
        running_training_loss = loss_function(BS_ANN(Xtrain),Ytrain.unsqueeze(1))
        test_loss.append(running_test_loss.detach().numpy())
        training_loss.append(running_training_loss.detach().numpy())
        scheduler.step()
        print(f"Executing epoch {epoch} of {EPOCHS} ")

    print(f"generating Loss chart for {EPOCHS} epochs")
    plt.figure(figsize=(9,3))
    plt.plot(np.arange(1,EPOCHS+1,1),np.log(np.array(training_loss)),'r',label="Training loss")
    plt.plot(np.arange(1,EPOCHS+1,1),np.log(np.array(test_loss)),'b',linestyle='dashed',label="Test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Log MSE loss")
    plt.legend()
    #plt.show()

    print(f"calculating BS using generated training model")
    S = 100                         # Strike
    K = np.arange(25,990,1)
    tau = 1
    r = 0.05
    sigma = 0.5
    inputs = np.zeros((len(K),4))

    for i in range(len(K)):
        inputs[i,:] = [S/K[i],tau,r,sigma]
    inputs = torch.tensor(inputs).type(torch.float)
    real = bs(S,K,tau,r,sigma)
    result = BS_ANN(inputs).detach()
    torch.save(BS_ANN.state_dict(), 'bs_model_weights.pth')

    plt.plot(K,real,'r',label="Real BS function")
    plt.plot(K,result,'b',linestyle='dashed',label="BS-ANN")
    plt.xlabel("Strike K")
    plt.ylabel("Scaled price V/K")
    plt.legend()
    plt.show()
#
# Run BS comparison
#
def run_bs_compare():
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

def runHeston_compare():
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

    H_param_space = np.array([M_H,TAU_H,MRS,R_H,VOLVOL,V_BAR,RHO,SIGMA_H])
    sampling_H = LHS(xlimits=H_param_space)
    num_H = 10**6
    # num_H = 1000
    x_H = sampling_H(num_H)

    labeled_H = np.zeros((num_H,9))
    for i in range(num_H):
        m,tau,mrs,r,volvol,v_bar,rho,sigma = x_H[i,0],x_H[i,1],x_H[i,2],x_H[i,3],x_H[i,4],x_H[i,5],x_H[i,6],x_H[i,7]
        price = Heston_2(S,S/m,tau,mrs,r,volvol,v_bar,rho,sigma,L,N)
        labeled_H[i,:] = [m,tau,mrs,r,volvol,v_bar,rho,sigma,price]

    H_data = torch.tensor(labeled_H).type(torch.float)

    # torch.save(H_data, 'H_data.pt')

def main():
    run_bs()

if __name__ == "__main__":
    print("Running BS ANN ...")
    main()
