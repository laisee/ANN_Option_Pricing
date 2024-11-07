import streamlit as st
import numpy as np
from scipy.stats import norm

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

# Example usage
S = 100  # Current asset price
K = 105  # Strike price
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility of the underlying asset
T = 1  # Time to maturity in years

call_price = binomial_tree(S, K, r, sigma, T, option_type='call')
put_price = binomial_tree(S, K, r, sigma, T, option_type='put')

print("Call Option Price:", call_price)
print("Put Option Price:", put_price)


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

st.markdown("<span style='font-size: 36px'>Power.Trade - Option Pricer</span>", unsafe_allow_html=True)
st.write("\n")
status_placeholder = st.empty()
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
        model_type = st.radio("Select Model", ["BinomialTree", "BlackScholes", "NeuralNetBS"], horizontal=False)
    with colInput:
        st.write("\n")
        spot_price = st.slider("Input Spot Price",0.10, 100.00, 5.0,0.1)
        strike = st.slider("Input Strike",0.10, 100.00, 5.00, 0.1)
        volatility = st.slider("Input Volatility",5.00, 100.00, 20.00, 1.0)
        riskless = st.slider("Input Riskless Rate",1.0, 10.00, 0.1, 0.1)
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
            """.format(option_type, spot_price, strike, maturity, volatility, riskless), unsafe_allow_html=True)
            st.write("\n")
            st.markdown("<h4><b><i>Price:</i></b></h4>", unsafe_allow_html=True)
            st.write("\n")
            if model_type.lower() == "binomialtree":
                option_price = binomial_tree(spot_price,strike,riskless,volatility/100.00,maturity,option_type,100)
                print(f"Option price[Binomial]: {option_price}")
            elif model_type.lower() == "blackscholes":
                option_price = black_scholes(spot_price,strike,maturity,riskless,volatility/100.00,option_type)
                print(f"Option price[BS]: {option_price}")
            elif model_type.lower() == "neuralnetbs":
                option_price = "To Be Completed"  # black_scholes(spot_price,strike,maturity,riskless,volatility,option_type)
                print(f"Option price[ANN_BS]: {option_price}")
            else:
                st.write(f"Invalid Model selection: {model_type}")
                option_price = -0.01
                print(f"Option price[err]: {option_price}")
            st.markdown("<h4><b><i>{}</i></b></h4>".format(option_price), unsafe_allow_html=True)
            st.write("\n")