# Generate training data for option pricing models
import numpy as np
from scipy.stats import norm
from smt.sampling_methods import LHS
import torch
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

def black_scholes(S, K, T, r, sigma, q=0, option_type="call"):
    """
    Calculate option price using Black-Scholes formula
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    q : float
        Dividend yield
    option_type : str
        Type of option ('call' or 'put')
        
    Returns:
    --------
    float
        Option price
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError(f"Invalid option type: {option_type}")
    
    return price

def generate_bs_training_data(num_samples=10**6, save_path="models/bs_data.pt"):
    """
    Generate training data for Black-Scholes model using Latin Hypercube Sampling
    
    Parameters:
    -----------
    num_samples : int
        Number of samples to generate
    save_path : str
        Path to save the generated data
        
    Returns:
    --------
    torch.Tensor
        Generated training data
    """
    print(f"Generating {num_samples} Black-Scholes training samples...")
    
    S = 100  # Base stock price
    
    # Parameter ranges for Latin Hypercube Sampling
    M_BS = [0.4, 1.6]     # moneyness = S0/K
    TAU_BS = [0.2, 1.1]   # time to maturity
    R_BS = [0.02, 0.1]    # risk-free rate
    SIGMA_BS = [0.01, 1.0]  # volatility
    
    BS_param_space = np.array([M_BS, TAU_BS, R_BS, SIGMA_BS])
    
    # Generate samples using Latin Hypercube Sampling
    sampling_BS = LHS(xlimits=BS_param_space)
    x_BS = sampling_BS(num_samples)
    
    # Calculate option prices for each sample
    labeled_BS = np.zeros((num_samples, 5))
    for i in range(num_samples):
        m, tau, r, sigma = x_BS[i, 0], x_BS[i, 1], x_BS[i, 2], x_BS[i, 3]
        price = black_scholes(S, S/m, tau, r, sigma)
        labeled_BS[i, :] = [m, tau, r, sigma, price]
    
    # Convert to PyTorch tensor
    BS_data = torch.tensor(labeled_BS).type(torch.float)
    
    # Save the data
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(BS_data, save_path)
        print(f"Saved {num_samples} samples to {save_path}")
    
    return BS_data

if __name__ == "__main__":
    generate_bs_training_data()