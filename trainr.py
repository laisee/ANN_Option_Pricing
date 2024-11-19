from datetime import datetime
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, LeakyReLU  # LeakyReLU directly in keras.layers
from keras.utils import to_categorical
from scipy.stats import norm
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler

TRAINING_TESTING_RATIO = 0.8

#
# ANN Model Settings
#
NODES = 120
EPOCHS = 20
BATCH_SIZE = 64
LAYER = 100
FEATURES = 5

folder = "."
file = "pt_mm_data.csv"


def calculate_months_from_current_date(date_string):
    date_format = "%Y%m%d"
    target_date = datetime.strptime(date_string, date_format)
    current_date = datetime.now()
    delta = (current_date.year - target_date.year) * 12 + (current_date.month - target_date.month)
    return abs(delta)

def custom_activation(x):
    return tf.math.exp(x)  # Adjusted for compatibility with TensorFlow 2.x

def create_nn_model(dim: int):
    model = Sequential([
        Input(shape=(FEATURES,)),       # Set to 6 to match the input data's feature count
        Dense(BATCH_SIZE),
        Activation(custom_activation),  # Apply the custom activation here
        Dropout(0.25),
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

def main():
    print("Creating random data for sample set of Option prices")
    print("initiating random seed")
    np.random.seed(0)  # for reproducibility
    
    print("Loading MM data ...")
    df = pd.read_csv(f"{folder}/{file}")

    # apply datetime format, assign zero where price missing
    df['timestamp'] = pd.to_datetime(df['TS'], unit='ns').dt.strftime('%Y-%m-%d')
    df["Ask"] = df['Ask'].replace('NAN', 0.00).astype(float)
    df["Bid"] = df['Bid'].replace('NAN', 0.00).astype(float)

    print("Calculate BS option prices")
    #
    # generate a set of option prices using MM data
    #
    #option_prices = [
    #    black_scholes(S, K, T, r, sigma, q) for S, K, T, r, sigma, q in zip(asset_prices, strike_prices, time_to_expiry, rates, volatility, dividends)
    #]
    # Coin,Call_Put,Product,Strike,Index_Price,Ask,Bid,Rate,TS
    df["Maturity"] = 3
    df["IV"] = 0.25
    df['Price'] = df.apply(lambda row: black_scholes(row['Index_Price'], row['Strike'], row["Maturity"], row['Rate'], row['IV'], 0, row['Call_Put']), axis=1)
    print(f"completed option price calculations for {df.shape[0]}")

    # 
    # Scale asset & option price equally since BS is linear homogenous in stock price  & strike price
    # see https://srdas.github.io/DLBook/DeepLearningWithPython.html#ref-CulkinDas
    #
    df["Index_price"] = df["Index_Price"]/df["Strike"]
    df["Price"] = df["Price"]/df["Strike"]

    n = df.shape[0]
    n_train =  (int)(TRAINING_TESTING_RATIO * n)
    # add df with training data from sample set
    train = df[0:n_train]

    print("load training dataset")
    X_train = train[['Index_Price', 'Strike', 'Maturity', 'IV', 'Rate' ]].values
    Y_train = train['Price'].values
    Y_train = Y_train.reshape(-1, 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    Y_train_scaled = scaler.fit_transform(Y_train)

    # add df with testing data from sample set, size is 1 - TRAINING_TESTING_RATIO X SAMPLE_SIZE
    test = df[n_train+1:n]
    X_test = test[['Index_Price', 'Strike', 'Maturity', 'IV', 'Rate']].values
    Y_test = test['Price'].values
    Y_test = Y_test.reshape(-1, 1)
    X_test_scaled = scaler.fit_transform(X_test)
    Y_test_scaled = scaler.fit_transform(Y_test)

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
    #show_charts(Y_train_scaled, Y_train_hat, stats)

if __name__ == "__main__":
    print("Running BS ANN(MM data) ...")
    main()
