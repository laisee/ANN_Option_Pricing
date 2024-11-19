#
# taken from https://blog.quantinsti.com/heston-model/
#
import numpy as np
from scipy.integrate import quad

# Model parameters
S0 = 100.0    # Initial stock price
K = 100.0     # Strike price
r = 0.05      # Risk-free rate
T = 1.0       # Time to maturity
kappa = 2.0   # Mean reversion rate
theta = 0.05  # Long-term average volatility
sigma = 0.3   # Volatility of volatility
rho = -0.5    # Correlation coefficient
v0 = 0.05     # Initial volatility

# Define characteristic functions
def heston_characteristic_function(u, S0, K, r, T, kappa, theta, sigma, rho, v0):
   xi = kappa - rho * sigma * 1j * u
   d = np.sqrt((rho * sigma * 1j * u - xi)**2 - sigma**2 * (-u * 1j - u**2))
   g = (xi - rho * sigma * 1j * u - d) / (xi - rho * sigma * 1j * u + d)
   C = r * 1j * u * T + (kappa * theta) / sigma**2 * ((xi - rho * sigma * 1j * u - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
   D = (xi - rho * sigma * 1j * u - d) / sigma**2 * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
   return np.exp(C + D * v0 + 1j * u * np.log(S0))

# Define functions to compute call and put options prices
def heston_call_price(S0, K, r, T, kappa, theta, sigma, rho, v0):
   integrand = lambda u: np.real(np.exp(-1j * u * np.log(K)) / (1j * u) * heston_characteristic_function(u - 1j, S0, K, r, T, kappa, theta, sigma, rho, v0))
   integral, _ = quad(integrand, 0, np.inf)
   return np.exp(-r * T) * 0.5 * S0 - np.exp(-r * T) / np.pi * integral


def heston_put_price(S0, K, r, T, kappa, theta, sigma, rho, v0):
   integrand = lambda u: np.real(np.exp(-1j * u * np.log(K)) / (1j * u) * heston_characteristic_function(u - 1j, S0, K, r, T, kappa, theta, sigma, rho, v0))
   integral, _ = quad(integrand, 0, np.inf)
   return np.exp(-r * T) / np.pi * integral - S0 + K * np.exp(-r * T)

# Calculate call and put option prices
call_price = heston_call_price(S0, K, r, T, kappa, theta, sigma, rho, v0)
put_price = heston_put_price(S0, K, r, T, kappa, theta, sigma, rho, v0)

print("Params:\n")
print(f"\tAsset:     {S0}")
print(f"\tStrike:    {K}")
print(f"\tRate:      {r}")
print(f"\tMaturity:  {T}")
print(f"\tKappa:     {kappa}")
print(f"\tTheta:     {theta}")
print(f"\tVol/sigma: {sigma}")
print(f"\tRho:       {rho}")
print(f"\tInit Vol:  {v0}")

print("\n\n")
print("Euro Call Price:", np.round(call_price, 4))
print("Euro Put  Price:", np.round(put_price, 4))
print("\n")
