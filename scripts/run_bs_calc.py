import sys
import os

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from models.blackscholes import bs as BS

S = 1000.0  # Asset price
K = 1000.0  # Strike price
T = 1.0  # 1 Year to maturity
R = 0.05  # 5% Risk-free rate
sigma = 0.15  # 15% Volatility

print("\nCalc BS for Call, Put using python 'blackscholes' library\n")
print(f"Params:\nAsset Price {S} Strike {K} Maturity {T} Rate {R} Vol {sigma}")

call = BS(S, K, T, R, sigma, "call")
print("\nPrice\nCall: ", call)

put = BS(S, K, T, R, sigma, "put")
print("Put:  ",put)
