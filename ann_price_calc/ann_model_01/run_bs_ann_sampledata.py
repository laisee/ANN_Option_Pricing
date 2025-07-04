from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, LeakyReLU  # LeakyReLU directly in keras.layers
import numpy as np
import pandas as pd
from scipy.stats import norm
import tensorflow as tf
import psutil

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler

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
EPOCHS = 20
BATCH_SIZE = 64
LAYER = 100
FEATURES = 6

def checkAccuracy(y,y_hat):
    print(f"Checking accuracy for y {len(y)} vs y_hat {len(y_hat)}")
    try:
        stats = dict()
        try:
            print("Calc y - y_hat")
            print(f"1Memory usage before calculation: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
 
            y = np.ravel(y)
            y_hat = np.ravel(y_hat)

            # Create a DataFrame with y and y_hat
            data = pd.DataFrame({'y': y, 'y_hat': y_hat})
        
            # Calculate the difference efficiently using pandas
            data['diff'] = data['y'] - data['y_hat']
            print(f"1Memory usage after calculation:  {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")

            # Store the 'diff' column in the stats dictionary
            stats['diff'] = data['diff']
            print("Calc'd y - y_hat")
            print(f"1Memory usage after assigning 'diff' to stats: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")

            print(f"2Memory usage before calculation: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
            print("Calc MSE**2 ...")
            stats['mse'] = np.mean(stats['diff']**2)
            print(f"2Memory usage after calculation:  {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
            print("Mean Squared Error:      ", stats['mse'])
    
            print("Calc RMSE ...")
            print(f"3Memory usage before calculation: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
            stats['rmse'] = np.sqrt(stats['mse'])
            print(f"3Memory usage after calculation:  {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
            print("Root Mean Squared Error: ", stats['rmse'])
    
            print("Calc MAE ...")
            stats['mae'] = np.mean(abs(stats['diff']))
            print("Mean Absolute Error:     ", stats['mae'])
        except Exception as e:
            print(f"Error occurred: {e}")
        print("Calc MPE ...")
        stats['mpe'] = np.sqrt(stats['mse'])/np.mean(y)
        print("Mean Percent Error:      ", stats['mpe'])
        return stats
    except Exception as e:
        print(f"Error while checking accuracy: {e}") 

def show_charts(y, y_hat, stats) -> None:
    #plots
    mpl.rcParams['agg.path.chunksize'] = 100000
    plt.figure(figsize=(14,10))
    plt.scatter(y, y_hat,color='black',linewidth=0.3,alpha=0.4, s=0.5)
    plt.xlabel('Actual Price',fontsize=20,fontname='Times New Roman')
    plt.ylabel('Predicted Price',fontsize=20,fontname='Times New Roman') 
    plt.show()
    
    stats['diff'] = np.array(stats['diff']).flatten()  # Flatten to make sure it's 1D
    plt.figure(figsize=(14,10))
    plt.hist(stats['diff'], bins=50,edgecolor='black',color='white')
    plt.xlabel('Diff')
    plt.ylabel('Density')
    plt.show()

def custom_activation(x):
    return tf.math.exp(x)  # Adjusted for compatibility with TensorFlow 2.x

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
    return price

def create_nn_model(dim: int):
    model = Sequential([
        Input(shape=(FEATURES,)),       # Set to 6 to match the input data's feature count
        Dense(BATCH_SIZE),
        Activation(custom_activation),  # Apply the custom activation here
        Dropout(0.25),
        Dense(1)
    ])
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
    model.compile(loss='mse',optimizer='adam')
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
    asset_prices = np.random.uniform(0.01, 95000.00, n_samples)   # Stock price between 0.01 and 75000.00
    strike_prices = np.random.uniform(1.00, 95000.00, n_samples)  # Strike price between 0.01 and 75000.00
    time_to_expiry = np.random.uniform(0.1, 2, n_samples)  # Expiry between 0.1 and 2 years
    dividends = np.random.uniform(0, 0.01, n_samples)      # Dividend yield between 0 and 3%
    volatility = np.random.uniform(0.01, 1.5, n_samples)   # Volatility between 1% and 150%
    rates = np.random.uniform(0.01, 0.05, n_samples)       # Risk-free rate between 1% and 5%

    print("calculate option prices")
    #
    # generate a set of option prices using randomized sample data
    #
    option_prices = [
        black_scholes(S, K, T, r, sigma, q) for S, K, T, r, sigma, q in zip(asset_prices, strike_prices, time_to_expiry, rates, volatility, dividends)
    ]
    print(f"completed option price calculations for {len(option_prices)}")

    print("copy options data into dict for loading into pandas dataframe")
    data = {
        "asset_price": asset_prices,
        "expiry": time_to_expiry,
        "dividends": dividends,
        "option_price": option_prices,
        "rate": rates,
        "strike": strike_prices,
        "volatility": volatility,
    }

    df = pd.DataFrame(data)
    print("loaded options data into pandas dataframe")

    # 
    # Scale asset & option price equally since BS is linear homogenous in stock price  & strike price
    # see https://srdas.github.io/DLBook/DeepLearningWithPython.html#ref-CulkinDas
    #
    #df["asset_price"] = df["asset_price"]/df["strike"]
    #df["option_price"] = df["option_price"]/df["strike"]

    n = SAMPLE_SIZE
    n_train =  (int)(TRAINING_TESTING_RATIO * n)
    # add df with training data from sample set
    train = df[0:n_train]

    print("load training dataset")
    X_train = train[['asset_price', 'strike', 'expiry', 'dividends', 'volatility', 'rate' ]].values
    Y_train = train['option_price'].values
    Y_train = Y_train.reshape(-1, 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    Y_train_scaled = scaler.fit_transform(Y_train)

    # Check if X_train is a numpy array
    #if isinstance(X_train_scaled, np.ndarray):
    #    X_train_scaled = X_train_scaled.astype(np.float32)  # Directly convert to float32
    #else:
    #    X_train_scaled = np.array(X_train_scaled, dtype=np.float32)  # Convert from other types

    # Similarly handle Y_train
    #if isinstance(Y_train_scaled, np.ndarray):
    #    X_train_scaled = X_train_scaled.astype(np.float32)  # Directly convert to float32
    #else:
    #    X_train_scaled = np.array(X_train_scaled, dtype=np.float32)  # Convert from other types

    # add df with testing data from sample set, size is 1 - TRAINING_TESTING_RATIO X SAMPLE_SIZE
    test = df[n_train+1:n]
    X_test = test[['asset_price', 'strike', 'expiry', 'dividends', 'volatility', 'rate']].values
    Y_test = test['option_price'].values
    Y_test = Y_test.reshape(-1, 1)
    scaler.fit_transform(X_test)
    scaler.fit_transform(Y_test)

    # Check if X_train is a numpy array
    #if isinstance(X_test_scaled, np.ndarray):
    #    X_test_scaled = X_test_scaled.astype(np.float32)  # Directly convert to float32
    #else:
    #    X_test_scaled = np.array(X_test_scaled, dtype=np.float32)  # Convert from other types

    # Similarly handle Y_train
    #if isinstance(Y_test_scaled, np.ndarray):
    #    Y_test_scaled = Y_test_scaled.astype(np.float32)  # Directly convert to float32
    #else:
    #    Y_test_scaled = np.array(Y_test_scaled, dtype=np.float32)  # Convert from other types

    #
    # generate ANN model with an Input layer
    #
    model = create_nn_model(X_train_scaled.shape[1])

    #
    # Fit the model to the Training data
    #           
    model.fit(X_train_scaled, Y_train_scaled, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=2)

    #
    # Calculate [prediction using trained model]
    #           
    Y_train_hat = model.predict(X_train_scaled)

    #
    # Reduce dim (240000,1) -> (240000,) to match Y_train's dim
    #
    Y_train_hat = np.squeeze(Y_train_hat)

    #
    # Calc stats and print errors
    #
    stats = checkAccuracy(Y_train_scaled, Y_train_hat)
    print(f"Stats: {stats}")
    #
    # Display charts (accuracy, pde)
    #
    show_charts(Y_train_scaled, Y_train_hat, stats)

if __name__ == "__main__":
    print("Running BS ANN(sample data) ...")
    main()
