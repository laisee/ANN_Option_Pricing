import streamlit as st
from streamlit_autorefresh import st_autorefresh
import numpy as np
from random import randint
import torch
import json
import redis
import pandas as pd
import datetime as dt
import os
from scipy.integrate import quad
from dotenv import load_dotenv
from scipy.stats import norm
from models.blackscholes import BlackScholes_ANN as bs_ann

# Hestonn model parameters
kappa = 2.0   # Mean reversion rate
theta = 0.05  # Long-term average volatility
sigma = 0.3   # Volatility of volatility
rho = -0.5    # Correlation coefficient
v0 = 0.05     # Initial volatility

COINS = ["ADA", "ARB", "BTC", "DOGE", "DOT", "ETH", "SOL", "TON", "TRX10000", "XRP"]
# Set the interval in milliseconds (e.g., 2000 ms = 2 seconds)
st.set_page_config(page_title='NN OptionPricer', layout="wide")
st_autorefresh(interval=75000, key="auto_refresh")

# Initialize the model
model = bs_ann()

# Load the saved weights
path = "models/bs_model_weights.pth"
model.load_state_dict(torch.load(path, weights_only=True))

if 'coin' not in st.session_state:
    st.session_state.coin = COINS[0]
if 'expiry' not in st.session_state:
    st.session_state.expiry = '27-Dec-24'
if "text" not in st.session_state:
    st.session_state.text = ''
if 'results' not in st.session_state:
    st.session_state.results = pd.DataFrame()

load_dotenv()

def get_redis_connection(host: str, pwd: str, port: int):
    return redis.Redis(host=host, port=port, password=pwd, db=0, socket_timeout=5, decode_responses=True )
def get_quotes(pattern: str, coin: str, count: int = 100) -> dict:
    quotes = []
    conn = get_redis_connection(os.getenv('REDIS_HOST'),os.getenv('REDIS_PWD'),os.getenv('REDIS_PORT'))
    cursor = '0'
    while cursor != 0:
        cursor, keys = conn.scan(cursor=cursor, match=pattern, count=count)
        if keys:
            # Use a pipeline to fetch all key values in a single round-trip
            with conn.pipeline() as pipe:
                for key in keys:
                    print( f"getting record '{key}'")
                    pipe.get(key)
                values = pipe.execute()
            for key, value in zip(keys, values):
                if key is not None and value is not None:
                    try:
                        obj = json.loads(value)
                        obj["strike"] = key.replace(":P", "").replace(":C", "").split("-")[2]
                        obj["type"] = "Call" if key.endswith("C") else "Put"
                        obj['bid_price'] = convert(coin, obj['bid_price'])
                        obj['ask_price'] = convert(coin, obj['ask_price'])
                        quotes.append(json.dumps(obj))
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
    st.session_state.text = f"completed query for {pattern}, found {len(quotes)} records"
    print(f"completed query for {pattern}, found {len(quotes)} records")
    return quotes
def binomial_tree(S, K, r, sigma, T, option_type, n=100):
    """
    Calculate the price of a European call or put option using a Binomial Tree.

    Parameters:
    S (float): Current asset price.
    K (float): Strike price.
    r (float): Risk-free interest rate (annualized).
    sigma (float): Volatility of the underlying asset (annualized).
    T (float): Time to maturity (years).
    n (int): Number of steps in the binomial tree.
    option_type (str): Type of option ('call' or 'put').

    Returns:
    float: Option price.
    """
    # Calculate time step and up/down factors
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

    # Initialize asset price tree
    asset_prices = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            asset_prices[i, j] = S * (u ** j) * (d ** (i - j))

    # Calculate option payoff at maturity
    if option_type.lower() == 'call':
        option_payoffs = np.maximum(asset_prices[-1, :] - K, 0)
    elif option_type.lower() == 'put':
        option_payoffs = np.maximum(K - asset_prices[-1, :], 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Calculate option price by backward induction
    option_prices = np.zeros((n + 1, n + 1))
    option_prices[-1, :] = option_payoffs
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_prices[i, j] = np.exp(-r * dt) * (p * option_prices[i + 1, j + 1] + (1 - p) * option_prices[i + 1, j])

    return option_prices[0, 0]
def black_scholes(S, K, T, r, sigma, option_type="call") -> float:
    price = -1.00
    q = 0.01
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError(f"error - invalid option type: {option_type}")
    return price
def monte_carlo( asset_price, strike, risk_free_rate, volatility, maturity, option_type, n=10000):
    # Calculate the drift and diffusion components
    dt = maturity
    drift = (risk_free_rate - (0.5 * volatility/100.00**2)) * dt
    print(f"MC:Drift: {drift}")
    diffusion = volatility * np.sqrt(dt)
    print(f"MC:Diffusion: {diffusion}")

    # Simulate asset price paths
    asset_prices_at_maturity = asset_price * np.exp(drift + diffusion * np.random.randn(n))

    # Calculate the option payoff at maturity
    if option_type.lower() == "call":
        payoffs = np.maximum(asset_prices_at_maturity - strike, 0)
    elif option_type.lower() == "put":
        payoffs = np.maximum(strike - asset_prices_at_maturity, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    print(f"MC:Payoffs: {payoffs}")

    # Discount the expected payoff back to present value
    option_price = np.exp(-risk_free_rate * maturity) * np.mean(payoffs)
    print(f"MC:Option Price: {option_price}")
    return option_price
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

date_conversion = {
    "11-Nov-24": "20241111",
    "12-Nov-24": "20241112",
    "13-Nov-24": "20241113",
    "14-Nov-24": "20241114",
    "15-Nov-24": "20241115",
    "16-Nov-24": "20241116",
    "17-Nov-24": "20241117",
    "18-Nov-24": "20241118",
    "19-Nov-24": "20241119",
    "20-Nov-24": "20241120",
    "21-Nov-24": "20241121",
    "22-Nov-24": "20241122",
    "29-Nov-24": "20241129",
    "27-Dec-24": "20241227",
    "06-Dec-24": "20241206",
    "28-Mar-25": "20250328",
    "27-Jun-25": "20250627",
    "28-Sep-25": "20250926"
}
st.markdown("<span style='font-size: 36px'>Power.Trade - Option Pricer</span>", unsafe_allow_html=True)
st.write("\n")
status_placeholder = st.empty()

tabIV, tabPrice = st.tabs(["Implied Vol","Price"])
def convert(coin, value, default=0.00):
    precision = {
        "BCH": 6,
        "BNB": 3,
        "BTC": 2,
        "DOGE":3,
        "ETH": 3,
        "FIL": 4,
        "ICP": 2,
        "LINK":2,
        "NEAR":2,
        "ORDI":3,
        "SOL": 4,
        "TON": 6,
        "XRP": 6
    }
    try:
        return float(value) / 10**precision.get(coin)
    except (TypeError, ValueError):
        return default
def getMarketPrice(coin: str, expiry: str):
    expiry = date_conversion.get(expiry)
    key = f"tob:{coin}-{expiry}*"
    print(f"KEY: {key}")
    quotes = get_quotes(key, coin)
    st.session_state.text = ""
    sorted_data = None
    if len(quotes) > 0:
        data = pd.DataFrame([json.loads(quote) for quote in quotes])
        data = data.drop(columns=["market_id", "timestamp", "tradeable_entity_id", "bid_quantity", "ask_quantity"])
        data = data.rename(columns={"symbol": "product", "bid_price": "bid", "ask_price": "ask"})
        sorted_data = data.sort_values(by=["type", "strike"])
    else:
        sorted_data = pd.DataFrame()
    st.session_state.results = sorted_data
    st.session_state.text = "updated @ ts = " + str(dt.datetime.now(dt.UTC))
# Define a callback function to update the text input based on radio selection
def update_selected_coin():
    st.session_state.text = f"You selected: {st.session_state.coin} with expiry '{st.session_state.expiry}'" if st.session_state.expiry is not None else  f"You selected: {st.session_state.coin}" 
    print(f"selected coin: {st.session_state.coin}")
    getMarketPrice(st.session_state.coin,  st.session_state.expiry)
def update_selected_expiry():
    st.session_state.text = f"You selected: {st.session_state.expiry} with coin '{st.session_state.coin}'" if st.session_state.coin is not None else  f"You selected: {st.session_state.expiry}" 
    print(f"selected expiry: {st.session_state.expiry}")
    getMarketPrice(st.session_state.coin,  st.session_state.expiry)
with tabIV:
    colInput, colSpacer, colResults = st.columns([0.30,0.05,0.65])
    with st.container():
        st.markdown(
            """
            <style>
            .main > div {
                max-width: 80%;
                margin: auto;
            }
            </style>
            """,
            unsafe_allow_html=True
        ) 
        with colInput:
            st.write("\n")
            coin = st.radio(
                "select Coin", 
                COINS,
                key='coin',
                on_change=update_selected_coin,
                horizontal=True)
            expiry = st.radio(
                "select Expiry", 
                ["12-Nov-24", "13-Nov-24", "14ov-24", "15-Nov-24", "22-Nov-24", "27-Dec-24", "25-Mar-25", "27-Jun-25", "28-Sep-25"], 
                key='expiry',
                on_change=update_selected_expiry,
                horizontal=False)
        with colSpacer:
            st.write("\n")
        with colResults:
            st.write("\n")
            st.text_input(
                '...',
                value=st.session_state.text,
                key="text"
            )
            st.dataframe(st.session_state.results, hide_index=True, column_order=["type","strike","symbol","bid", "ask"], use_container_width=True)
with tabPrice:
    colSetting, colInput, colSpacer, colCalc = st.columns([0.30, 0.20, 0.10, 0.45])
    st.markdown(
        """
        <style>
        .main > div {
            max-width: 80%;
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.container():
        st.markdown('<div class="container">', unsafe_allow_html=True)
        with colSetting:
            st.write("\n")
            option_type = st.radio("Select Option Type", ["Call", "Put"], horizontal=True)
            exercise_type = st.radio("Select Exercise Type", ["European", "American"], horizontal=True, disabled=True)
            st.write("\n")
            model_type = st.radio("Select Model", ["BinomialTree", "BlackScholes", "Heston", "NeuralNetBS"], horizontal=False)
        with colInput:
            st.write("\n")
            spot_price = st.slider("Input Spot Price",1000.00, 10000.00, 3100.0, 100.00)
            strike = st.slider("Input Strike",1000.00, 10000.00, 3000.00, 100.00)
            volatility = st.slider("Input Volatility",5.00, 100.00, 20.00, 1.0)
            riskfree = st.slider("Input Riskfree Rate",1.0, 10.00, 0.05, 0.1)
            maturity = st.slider("Select months till maturity",1, 18, 3, 1)
        with colSpacer:
            st.write("")
        with colCalc:
            option_price = 0.00
            st.markdown("""
            <style>
            .stButton > button,
            .stButton > button:hover,
            .stButton > button:focus,
            .stButton > button:active {
                color: #00ff00 !important;
                border-color: #00ff00 !important;
                background-color: #004400 !important;
                box-shadow: none !important;
                outline: none !important;
                margin: 0 auto;
            }
            </style>
            """, unsafe_allow_html=True)
            calc = st.button("Calculate Price")
            if calc:
                st.markdown("<h4><b><i>Parameters:</i></b></h4>", unsafe_allow_html=True)
                st.markdown("""
                    <table>
                        <tr>
                            <th>Type:</th>
                            <td align="right">{}</td>
                        </tr>
                        <tr>
                            <th>Spot:</th>
                            <td align="right">{}</td>
                        </tr>
                        <tr>
                            <th>Strike:</th>
                            <td align="right">{}</td>
                        </tr>
                        <tr>
                            <th>Expiry:</th>
                            <td align="right">{}</td>
                        </tr>
                        <tr>
                            <th>Vol:</th>
                            <td align="right">{}</td>
                        </tr>
                        <tr>
                            <th>Rate:</th>
                            <td align="right">{}</td>
                        </tr>
                    </table>
                """.format(option_type, spot_price, strike, maturity, volatility, riskfree), unsafe_allow_html=True)
                st.write("\n")
                st.markdown("<h4><b><i>Price:</i></b></h4>", unsafe_allow_html=True)
                st.write("\n")
                if model_type.lower() == "binomialtree":
                    option_price = binomial_tree(spot_price,strike,riskfree,volatility/100.00,maturity,option_type,100)
                    print(f"Option price[Binomial]: {option_price}")
                elif model_type.lower() == "blackscholes":
                    option_price = black_scholes(spot_price,strike,maturity,riskfree,volatility/100.00,option_type)
                    print(f"Option price[BS]: {option_price}")
                elif model_type.lower() == "montecarlo":
                    option_price = monte_carlo(spot_price, strike, riskfree, volatility, maturity, option_type)
                    print(f"Option price[Monte Carlo]: {option_price}")
                elif model_type.lower() == "heston":
                    # Calculate call and put option prices
                    if option_type.lower() == "call":
                        print(f"Heston/Call: {option_price}")
                        option_price = heston_call_price(spot_price, strike, riskfree, maturity, kappa, theta, sigma, rho, v0)
                    elif option_type.lower() == "put":
                        option_price = heston_put_price(spot_price, strike, riskfree, maturity, kappa, theta, sigma, rho, v0)
                        print(f"Heston/Put: {option_price}")
                elif model_type.lower() == "neuralnetbs":
                    q = 0.00
                    # Set the model to evaluation mode
                    model.eval()

                    inputs = [spot_price, maturity, q, volatility] # example: [stock price, time to expiry, dividends, volatility]
                    RATE = 0.05

                    # Adjust the shape according to your input dimensions.
                    sample_input = torch.tensor([inputs])

                    #S   = inputs[0]      # stock price
                    #K   = inputs[0] # strike
                    #t   = inputs[1]      # tau
                    #r   = RATE           # risk-free rate
                    #vol = inputs[3]      # vol

                    with torch.no_grad():
                        option_price = model(sample_input).item()
                    print(f"Option price[ANN_BS]: {option_price}")
                else:
                    st.write(f"Invalid Model selection: {model_type}")
                    option_price = -0.01
                    print(f"Option price[err]: {option_price}")
                st.markdown("<h4><b><i>{}</i></b></h4>".format(option_price), unsafe_allow_html=True)
                st.write("\n")
