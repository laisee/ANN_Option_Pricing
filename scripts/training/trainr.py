from scipy.stats import norm  # Required for norm.cdf
import numpy as np

# Black-Scholes formula for calculating option price
def black_scholes(S, K, T, r, sigma, q=0, option_type="call") -> float:
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError(f"error - invalid option type: {option_type}")
    
    print(f"Option price for Price/Strike {S}/{K} expiring at {T} year with Vol {sigma*100}% ->: {price}")
    return price

def main():
    # Input parameters
    S = 100.0        # Asset price
    K = 100.0        # Strike price
    T = 1.0          # Time to maturity (in years)
    r = 0.05         # Risk-free interest rate
    sigma = 0.2      # Volatility
    q = 0.0          # Dividend yield
    option_type = "call"

    # Compute option price
    price = black_scholes(S, K, T, r, sigma, q, option_type)
    print(f"Calculated {option_type} option price:", round(price,4))
    option_type = "put"
    price = black_scholes(S, K, T, r, sigma, q, option_type)
    print(f"Calculated {option_type} option price:", round(price,4))

if __name__ == "__main__":
    main()
