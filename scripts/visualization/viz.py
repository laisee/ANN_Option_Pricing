# Visualization utilities for option pricing models
import matplotlib.pyplot as plt

def plot_option_prices(strikes, prices, model_name="Model", option_type="call"):
    """
    Plot option prices across different strike prices
    
    Parameters:
    -----------
    strikes : array-like
        Array of strike prices
    prices : array-like
        Array of option prices corresponding to strikes
    model_name : str
        Name of the pricing model
    option_type : str
        Type of option ('call' or 'put')
    """
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, prices, 'b-', linewidth=2)
    plt.title(f"{option_type.capitalize()} Option Prices - {model_name}")
    plt.xlabel("Strike Price")
    plt.ylabel("Option Price")
    plt.grid(True)
    plt.show()

def compare_models(strikes, prices_dict, option_type="call"):
    """
    Compare option prices from different models
    
    Parameters:
    -----------
    strikes : array-like
        Array of strike prices
    prices_dict : dict
        Dictionary mapping model names to arrays of prices
    option_type : str
        Type of option ('call' or 'put')
    """
    plt.figure(figsize=(12, 8))
    
    for model_name, prices in prices_dict.items():
        plt.plot(strikes, prices, linewidth=2, label=model_name)
    
    plt.title(f"{option_type.capitalize()} Option Price Comparison")
    plt.xlabel("Strike Price")
    plt.ylabel("Option Price")
    plt.legend()
    plt.grid(True)
    plt.show()
