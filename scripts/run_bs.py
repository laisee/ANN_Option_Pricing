from blackscholes import BlackScholesCall, BlackScholesPut

S = 1000.0  # Asset price
K = 1000.0  # Strike price
T = 1.0  # 1 Year to maturity
R = 0.05  # 5% Risk-free rate
sigma = 0.15  # 15% Volatility
q = 0.0  # 0% Annual Dividend Yield

print("\nCalc BS for Call, Put using python 'blackscholes' library\n")
print(f"Params:\nAsset Price {S} Strike {K} Maturity {T} Rate {R} Vol {sigma}")

call = BlackScholesCall(S, K, T, R, sigma, q)
print("\nPrice\nCall: ",call.price())

put = BlackScholesPut(S, K, T, R, sigma, q)
print("Put:  ",put.price())
