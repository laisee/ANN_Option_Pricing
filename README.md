# Option Pricing using multiple pricing tools with Web App(streamlit)

[![Python App](https://github.com/laisee/ANN_Option_Pricing/actions/workflows/python-app.yml/badge.svg)](https://github.com/laisee/ANN_Option_Pricing/actions/workflows/python-app.yml)
[![Bandit Security Check](https://github.com/laisee/ANN_Option_Pricing/actions/workflows/main.yml/badge.svg)](https://github.com/laisee/ANN_Option_Pricing/actions/workflows/main.yml)
[![Dependabot Updates](https://github.com/laisee/ANN_Option_Pricing/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/laisee/ANN_Option_Pricing/actions/workflows/dependabot/dependabot-updates)

## Project Goals

This project provides a comprehensive platform for option pricing using various mathematical models and machine learning techniques. 
The key objectives are:

1. Implement multiple option pricing models:
   - Black-Scholes model (analytical solution)
   - Binomial Tree model (discrete-time approximation)
   - Heston model (stochastic volatility)
   - Neural Network models (machine learning approach)

2. Provide an interactive web interface for option pricing calculations

3. Demonstrate how neural networks can approximate complex financial models

4. Enable comparison between different pricing methodologies

5. Offer both code-based and pre-trained model approaches to option pricing

## LEARNING: Option Pricing Models

### Black-Scholes Model
The Black-Scholes model is a mathematical model for pricing options contracts. It assumes that the price of the underlying asset follows a geometric Brownian motion with constant volatility.

- Implementation: `models/blackscholes.py` and `ann_price_calc/ann_model_01/run_bs_ann_sampledata.py`
- Runner: `scripts/blackscholes/run_bs_calc.py`
- Key parameters: spot price, strike price, time to maturity, risk-free rate, volatility

### Binomial Tree Model
The Binomial Tree model is a discrete-time model that simulates the evolution of the underlying asset price through a lattice of possible price paths.

- Implementation: `models/binomial.py`
- Key parameters: spot price, strike price, time to maturity, risk-free rate, volatility, number of steps

### Heston Model
The Heston model extends the Black-Scholes model by allowing for stochastic volatility, which better captures market dynamics.

- Implementation: `models/heston.py`
- Runners: 
  - `scripts/heston/calc_heston.py` (standard implementation)
  - `scripts/heston/calc_heston2.py` (alternative implementation)
- Key parameters: spot price, strike price, time to maturity, risk-free rate, initial volatility, mean reversion rate, long-term volatility, volatility of volatility, correlation

### Neural Network Models
Neural networks can be trained to approximate option pricing functions, potentially offering faster computation for complex models.

- Implementation: 
  - `models/blackscholes.py` (BlackScholes_ANN class)
  - `models/tiny_model.py` (TinyModel class)
  - `ann_price_calc/original_ann/run_ann.py` (Original ANN implementation)
  - `ann_price_calc/ann_model_01/run_bs_ann_sampledata.py` (BS ANN with sample data)
  - `ann_price_calc/correlation/run_model.py` (Correlation network)
- Runners:
  - `scripts/utils/run_tinymodel.py`
  - `scripts/utils/exec_tinymodel.py`
  - `scripts/utils/simple_nn.py`

## TRAINING: Building Model Weights

### Training the Black-Scholes Neural Network

The Black-Scholes neural network model can be trained to approximate the Black-Scholes formula:

```
python scripts/blackscholes/run_model.py
```

Or using the package entry point:

```
python -m ann_price_calc.ann_model_01.run_bs_ann_sampledata
```

This will:
1. Generate training data using Latin Hypercube Sampling
2. Train the neural network model
3. Save the model weights to `models/bs_model_weights.pth`
4. Compare the neural network predictions with the analytical Black-Scholes formula

### Training the Heston Neural Network

The Heston neural network model can be trained to approximate the Heston model:

```
python scripts/heston/run_model.py
```

This will:
1. Generate training data for the Heston model
2. Train the neural network model
3. Save the model weights to `models/heston_model_weights.pth`

### Training the Correlation Network

The correlation network model can be trained using:

```
python -m ann_price_calc.correlation.run_model
```

This will:
1. Load correlation data from `ann_price_calc/correlation/correlation_data.csv`
2. Train a neural network to analyze correlations between assets
3. Save the model weights to `ann_price_calc/correlation/correlation_net_weights.pth`

### Training the Simple Neural Network

A simple neural network for binary classification can be trained using:

```
make run_simple
```

### Training the Tiny Model

The tiny model is a small neural network that can be trained using:

```
make run_tiny
```

This will save the model weights to `models/tinymodel.pth`.

## EXECUTION: Running the Application

### Local Development

Launch the Streamlit app locally:

```
streamlit run app.py
```

This will start a web server and open the application in your default browser. The application provides an interactive interface for:
- Setting option parameters (spot price, strike price, volatility, etc.)
- Selecting the pricing model
- Viewing option prices for both call and put options
- Comparing results across different models

### Running Individual Models

#### Black-Scholes Model
```
make run_bs_calc       # Run Black-Scholes calculation
```

#### Heston Model
```
make run_heston_calc   # Run Heston model calculation
make run_heston2_calc  # Run alternative Heston model calculation
```

#### Neural Network Models
```
make run_tiny          # Run tiny neural network model
make exec_tiny         # Execute tiny model with pre-trained weights
make run_simple        # Run simple neural network
```

#### ANN Package Models
```
python -m ann_price_calc.ann_model_01.run_bs_ann_sampledata  # Run BS ANN with sample data
python -m ann_price_calc.correlation.run_model               # Run correlation network
```

#### Visualization
```
make run_visual        # Run visualization tools
```

## DEPLOY: Deploying to Heroku

### Prerequisites
1. Heroku CLI installed
2. Logged in to Heroku (`heroku login`)
3. Access to a Heroku app

### Deployment Steps

1. Build and push the container to Heroku:
```
make build
```

2. Release the container on Heroku:
```
make release
```

3. Restart the application (if needed):
```
make restart
```

4. Stop the application (if needed):
```
make stop
```

### Deployment Configuration

The application is configured for Heroku deployment with:
- `Procfile` - Defines the command to run the application
- `setup.sh` - Sets up Streamlit configuration
- `Dockerfile` - Defines the container configuration

## Project Structure

- `app.py`: Main Streamlit application entry point
- `models/`: Model implementations and saved weights
  - `blackscholes.py`: Black-Scholes model and neural network implementation
  - `heston.py`: Heston model implementation
  - `tiny_model.py`: Tiny neural network model
- `scripts/`: Individual model runners and utilities
  - `blackscholes/`: Black-Scholes related scripts
  - `heston/`: Heston model related scripts
  - `utils/`: Utility scripts
  - `visualization/`: Visualization tools
- `ann_price_calc/`: Python package with ANN implementations
  - `ann_model_01/`: Black-Scholes ANN with sample data
  - `correlation/`: Correlation network and data
  - `original_ann/`: Original ANN implementation
- `data/`: Sample and training data

## Package Entry Points

The project provides several command-line entry points defined in `pyproject.toml`:

```
run-ann-model-01  # Run Black-Scholes ANN with sample data
run-ann-model-02  # Run ANN model 02
run-ann-model-03  # Run correlation network
```

These can be used after installing the package with `pip install -e .`
