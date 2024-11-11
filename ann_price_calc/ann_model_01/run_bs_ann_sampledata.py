from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, LeakyReLU  # LeakyReLU directly in keras.layers
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from scipy.stats import norm
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib as mpl

#
# Code taken from https://srdas.github.io/DLBook/DeepLearningWithPython.html#ref-CulkinDas
#

# 
# Runtime parameters
#
SAMPLE_SIZE = 300_000
TRAINING_TESTING_RATIO = 0.8

#
# ANN Model Settings
#
NODES = 120
EPOCHS = 10
BATCH_SIZE = 64
LAYER = 100
FEATURES = 6

def checkAccuracy(y,y_hat):
    print("Checking accuracy ...")
    stats = dict()
    stats['diff'] = y - y_hat
    
    print("calc MSE**2 ...")
    stats['mse'] = np.mean(stats['diff']**2)
    print("Mean Squared Error:      ", stats['mse'])
    
    print("calc RMSE ...")
    stats['rmse'] = np.sqrt(stats['mse'])
    print("Root Mean Squared Error: ", stats['rmse'])
    
    print("calc MAE ...")
    stats['mae'] = np.mean(abs(stats['diff']))
    print("Mean Absolute Error:     ", stats['mae'])
    
    print("calc MPE ...")
    stats['mpe'] = np.sqrt(stats['mse'])/np.mean(y)
    print("Mean Percent Error:      ", stats['mpe'])
    return stats

def show_charts(y, y_hat, stats) -> None:
    #plots
    mpl.rcParams['agg.path.chunksize'] = 100000
    plt.figure(figsize=(14,10))
    plt.scatter(y, y_hat,color='black',linewidth=0.3,alpha=0.4, s=0.5)
    plt.xlabel('Actual Price',fontsize=20,fontname='Times New Roman')
    plt.ylabel('Predicted Price',fontsize=20,fontname='Times New Roman') 
    plt.show()
    
    plt.figure(figsize=(14,10))
    plt.hist(stats['diff'], bins=50,edgecolor='black',color='white')
    plt.xlabel('Diff')
    plt.ylabel('Density')
    plt.show()

def custom_activation(x):
    return tf.math.exp(x)  # Adjusted for compatibility with TensorFlow 2.x

#
# Define the Black-Scholes formula function
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
    return price

def create_nn_model(dim: int):
    model = Sequential([
        Input(shape=(FEATURES,)),  # Set to 6 to match the input data's feature count
        Dense(BATCH_SIZE),
        Activation(custom_activation),  # Apply the custom activation here
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    model.add(Dense(NODES, input_dim=dim))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))

    model.add(Dense(NODES, activation='elu'))
    model.add(Dropout(0.25))

    model.add(Dense(NODES, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(NODES, activation='elu'))
    model.add(Dropout(0.25))

    model.add(Dense(1))
    model.add(Activation(custom_activation))
    #
    # Compile and build model
    #           
    model.compile(loss='mse',optimizer='rmsprop')
    return model

def price(
        asset_price=100.00,
        strike=100.00,
        expiry=0.5,
        rate=0.05,
        vol=0.20,
        call_put="call"
    ) -> float:
    return black_scholes(
        asset_price,
        strike,
        expiry,
        rate,
        vol,
        0,
        call_put
        )

def main():
    print("Creating random data for sample set of Option prices")
    print("initiating random seed")
    np.random.seed(0)  # for reproducibility

    print("generating random data")
    n_samples = SAMPLE_SIZE
    asset_prices = np.random.uniform(50, 150, n_samples)   # Stock price between 50 and 150
    strike_prices = np.random.uniform(50, 150, n_samples)  # Strike price between 50 and 150
    time_to_expiry = np.random.uniform(0.1, 2, n_samples)  # Expiry between 0.1 and 2 years
    dividends = np.random.uniform(0, 0.03, n_samples)      # Dividend yield between 0 and 3%
    volatility = np.random.uniform(0.1, 1.5, n_samples)    # Volatility between 10% and 150%
    rates = np.random.uniform(0.01, 0.05, n_samples)       # Risk-free rate between 1% and 5%

    print("calculate option prices")
    #
    # generate a set of option prices using randomized sample data
    #
    option_prices = [
        black_scholes(S, K, T, r, sigma, q)
        for S, K, T, r, sigma, q in zip(asset_prices, strike_prices, time_to_expiry, rates, volatility, dividends)
    ]
    print(f"completed option price calculations for {len(option_prices)}")

    print("copy options data into dict for loading into pandas dataframe")
    data = {
        "asset_price": asset_prices,
        "strike": strike_prices,
        "expiry": time_to_expiry,
        "dividend": dividends,
        "volatility": volatility,
        "rate": rates,
        "option_price": option_prices,
    }

    df = pd.DataFrame(data)
    print("loaded options data into pandas dataframe")

    # 
    # Scale asset & option price equally since BS is linear homogenous in stock price  & strike price
    # see https://srdas.github.io/DLBook/DeepLearningWithPython.html#ref-CulkinDas
    #
    df["asset_price"] = df["asset_price"]/df["strike"]
    df["option_price"] = df["option_price"]/df["strike"]

    n = SAMPLE_SIZE
    n_train =  (int)(TRAINING_TESTING_RATIO * n)
    # add df with training data from sample set
    train = df[0:n_train]

    print("load training dataset")
    X_train = train[['asset_price','strike', 'expiry', 'dividend', 'volatility', 'rate']].values
    Y_train = train['option_price'].values

    # Check if X_train is a numpy array
    if isinstance(X_train, np.ndarray):
        X_train = X_train.astype(np.float32)  # Directly convert to float32
    else:
        X_train = np.array(X_train, dtype=np.float32)  # Convert from other types

    # Similarly handle Y_train
    if isinstance(Y_train, np.ndarray):
        Y_train = Y_train.astype(np.float32)  # Directly convert to float32
    else:
        Y_train = np.array(Y_train, dtype=np.float32)  # Convert from other types

    print("load test dataset")
    # add df with testing data from sample set, size is 1 - TRAINING_TESTING_RATIO X SAMPLE_SIZE
    test = df[n_train+1:n]
    X_test = test[['asset_price','strike', 'expiry', 'dividend', 'volatility', 'rate']].values
    Y_test = test['option_price'].values

    # Check if X_train is a numpy array
    if isinstance(X_test, np.ndarray):
        X_test = X_test.astype(np.float32)  # Directly convert to float32
    else:
        X_test = np.array(X_test, dtype=np.float32)  # Convert from other types

    # Similarly handle Y_train
    if isinstance(Y_test, np.ndarray):
        Y_test = Y_test.astype(np.float32)  # Directly convert to float32
    else:
        Y_test = np.array(Y_test, dtype=np.float32)  # Convert from other types

    #
    # generate ANN model with an Input layer
    #
    model = create_nn_model(X_train.shape[1])

    #
    # Fit the model to the Training data
    #           
    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=2)

    #
    # Calculcate [prediction using trained model]
    #           
    Y_train_hat = model.predict(X_train)

    #
    # Reduce dim (240000,1) -> (240000,) to match Y_train's dim
    #
    Y_train_hat = np.squeeze(Y_train_hat)

    #
    # Calc stats and print errors
    #
    stats = checkAccuracy(Y_train, Y_train_hat)
    print(f"Stats: {stats}")
    #
    # Display charts (accuracy, pde)
    #
    #show_charts(Y_train, Y_train_hat, stats)

if __name__ == "__main__":
    print("Running BS ANN(sample data) ...")
    main()
