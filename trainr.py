from datetime import datetime
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, LeakyReLU  # LeakyReLU directly in keras.layers
from keras.optimizers import Adam, RMSprop
from scipy.stats import norm
import tensorflow as tf
import psutil

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler

TRAINING_TESTING_RATIO = 0.8

#
# ANN Model Settings
#
DATASET_SIZE=3148000
NODES = 120
EPOCHS = 20
BATCH_SIZE = 64
LAYER = 100
FEATURES = 5

folder = "."
file = "pt_mm_data.csv"

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

def checkAccuracy(y,y_hat):
    print(f"Checking accuracy for y {len(y)} vs y_hat {len(y_hat)}")
    stats = dict()
    try:
        print(f"1Memory usage before calculation: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")

        y = np.ravel(y)
        y_hat = np.ravel(y_hat)

        print(f"Memory usage b4 calculation:  {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
        # Create a DataFrame with y and y_hat
        data = pd.DataFrame({'y': y, 'y_hat': y_hat})

        # Calculate the difference efficiently using pandas
        data['diff'] = data['y'] - data['y_hat']

        # Store the 'diff' column in the stats dictionary
        stats['diff'] = data['diff']

        stats['mse'] = np.mean(stats['diff']**2)
        print("Mean Squared Error:      ", stats['mse'])

        stats['rmse'] = np.sqrt(stats['mse'])
        print("Root Mean Squared Error: ", stats['rmse'])

        stats['mae'] = np.mean(abs(stats['diff']))
        print("Mean Absolute Error:     ", stats['mae'])

        stats['mpe'] = np.sqrt(stats['mse'])/np.mean(y)
        print("Mean Percent Error:      ", stats['mpe'])
        print(f"Memory usage after calculation[mpe]:  {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")

    except Exception as e:
        print(f"Error while checking accuracy: {e}")
    return stats

def calculate_months_from_current_date(date_string):
    date_format = "%Y-%m-%d"
    target_date = datetime.strptime(date_string, date_format)
    current_date = datetime.now()
    delta = (current_date.year - target_date.year) * 12 + (current_date.month - target_date.month)
    return max(1,abs(delta))

def custom_activation(x):
    return tf.math.exp(x)  # Adjusted for compatibility with TensorFlow 2.x

def create_nn_model(dim: int):

    model = Sequential([
        Input(shape=(FEATURES,)),        # Set to the number of input features, e.g., 5
        Dense(BATCH_SIZE),               # Use an appropriate number of units for the Dense layer
        Activation(custom_activation),   # Apply the custom activation function
        Dropout(0.25),                   # Dropout for regularization
        Dense(1)                         # Output layer with one unit (for regression)
    ])
    Adam(learning_rate=0.0001)  # Example: Using 0.001 as the learning rate
    rms_opt  = RMSprop(learning_rate=0.0001)  # Example: Using 0.001 as the learning rate

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
    model.compile(optimizer=rms_opt, loss="mse")
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
    df = df.sample(n=DATASET_SIZE, random_state=42, replace=True)

    # BTC,Put,BTC-20241227-85000P,85000.0,90917.905,436200,543300,0.05,1731398698994917000
    df = df.rename(columns={
        df.columns[0]: 'coin',
        df.columns[1]: 'call_put',
        df.columns[2]: 'product',
        df.columns[3]: 'strike',
        df.columns[4]: 'index_price',
        df.columns[5]: 'bid',
        df.columns[6]: 'ask',
        df.columns[7]: 'rate',
        df.columns[8]: 'ts',
        })

    # apply datetime format, assign zero where price missing
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ns').dt.strftime('%Y-%m-%d')
    df["ask"] = df['ask'].replace('NAN', 0.00).astype(float)
    df["bid"] = df['bid'].replace('NAN', 0.00).astype(float)

    print("Calculate BS Option prices")
    #
    # generate a set of option prices using MM data
    #
    #option_prices = [
    #    black_scholes(S, K, T, r, sigma, q) for S, K, T, r, sigma, q in zip(asset_prices, strike_prices, time_to_expiry, rates, volatility, dividends)
    #]
    # Coin,Call_Put,Product,Strike,Index_Price,Ask,Bid,Rate,TS
    df["maturity"] = df.apply(lambda row: calculate_months_from_current_date(row["timestamp"]), axis=1)
    df["iv"] = 0.25
    df['price'] = df.apply(lambda row: black_scholes(row['index_price'], row['strike'], row["maturity"], row['rate'], row['iv'], 0, row['call_put']), axis=1)
    print(f"completed option price calculations for {df.shape[0]}")

    # 
    # Scale asset & option price equally since BS is linear homogenous in stock price  & strike price
    # see https://srdas.github.io/DLBook/DeepLearningWithPython.html#ref-CulkinDas
    #
    df["index_price"] = df["index_price"]/df["strike"]
    df["price"] = df["price"]/df["strike"]

    n = df.shape[0]
    n_train =  (int)(TRAINING_TESTING_RATIO * n)
    # add df with training data from sample set
    train = df[0:n_train]

    print("load training dataset")
    X_train = train[['index_price', 'strike', 'maturity', 'iv', 'rate']].values
    Y_train = train['price'].values
    Y_train = Y_train.reshape(-1, 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    Y_train_scaled = scaler.fit_transform(Y_train)

    # add df with testing data from sample set, size is 1 - TRAINING_TESTING_RATIO X SAMPLE_SIZE
    test = df[n_train+1:n]
    X_test = test[['index_price', 'strike', 'maturity', 'iv', 'rate']].values
    Y_test = test['price'].values
    Y_test = Y_test.reshape(-1, 1)
    scaler.fit_transform(X_test)
    scaler.fit_transform(Y_test)

    #
    # generate ANN model with an Input layer
    #
    model = create_nn_model(X_train_scaled.shape[1])
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    #torch.nn.init.xavier_uniform_(model.weights)

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
    show_charts(Y_train_scaled, Y_train_hat, stats)

    # save weights
    model.save_weights('models/pt_mm_model.weights.h5')

if __name__ == "__main__":
    print("Running BS ANN(MM data) ...")
    main()
