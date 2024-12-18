import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from scipy.stats import norm


BATCH_SIZE=64
FEATURES=5

#
# Black-Scholes formula for calculating option price
#
def black_scholes(S, K, T, r, sigma, q=0, option_type="call") -> float:
    price = -1.00
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError(f"error - invalid option type: {option_type}")
    print(f"Option price for Price/Strike {S}/{K} ->: {price}")
    return price

# Define the neural network model architecture
model = Sequential([
    Input(shape=(FEATURES,)),        # Set to the number of input features, e.g., 5
    Dense(BATCH_SIZE),               # Use an appropriate number of units for the Dense layer
    Dropout(0.25),                   # Dropout for regularization
    Dense(1)                         # Output layer with one unit (for regression)
])

# Load pre-trained weights
model.load_weights("models/pt_mm_model.weights.h5")

# Compile the model (required before making predictions)
model.compile(optimizer='adam', loss='mse')

# 'index_price', 'strike', 'maturity', 'iv', 'rate'
INDEX_PRICE=3150
STRIKE=3150
MATURITY=1.0
VOL=0.35
RATE=0.05
input_data = np.array([[INDEX_PRICE, STRIKE, MATURITY, VOL, RATE]])  # Replace with your data

prediction = model.predict(input_data)
bs = black_scholes(INDEX_PRICE, STRIKE, MATURITY, RATE, VOL)

print("NN: ", prediction[0][0])
print("BS: ", bs)
