import sys
import os

# Go up three directories from current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)

from models.blackscholes.blackscholes import bs as BS

S = 1000.0  # Asset price
K = 1000.0  # Strike price
T = 1.0     # Time to maturity (1 year)
R = 0.05    # Risk-free rate (5%)
sigma = 0.15  # Volatility (15%)

print("\nCalc BS for Call, Put using python 'blackscholes' library\n")
print(f"Params:\nAsset Price {S} Strike {K} Maturity {T} Rate {R} Vol {sigma}")

call = BS(S, K, T, R, sigma, "call")
print("\nPrice\nCall: ", call)

put = BS(S, K, T, R, sigma, "put")
print("Put:  ", put)