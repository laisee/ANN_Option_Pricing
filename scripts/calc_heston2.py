import math
import pandas as pd
import datetime, os
import numpy as np
import numpy.random as npr
from pylab import plt, mpl

import threading

from scipy.stats import norm
from scipy import optimize
import scipy.integrate as integrate
import scipy.special as special 

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

def heston_call(S, K, tau, r, kappa, theta, sigma, rho, nu, limit):
    X = np.log(S / K)
    kappahat = kappa - 0.5 * rho *sigma

    def Integrand(k):
        xi = np.sqrt(k**2*sigma**2*(1-rho**2) + 2j*k*sigma*rho*kappahat + \
                         kappahat**2 + (sigma**2)/4)
        psi_p = -(1j*k*rho*sigma + kappahat) + xi
        psi_m = (1j*k*rho*sigma + kappahat) + xi
        alpha = -((kappa*theta)/sigma**2) * \
              (psi_p*tau + 2*np.log((psi_m+psi_p*np.exp(-xi*tau))/(2*xi)))
        beta = (1-np.exp(-xi*tau)) / (psi_m + psi_p * np.exp(-xi*tau))

        numerator = np.exp((-1j*k+0.5)*X+alpha-(k**2+0.25)*beta*nu)
        integrall = np.real(numerator / (k**2+0.25))

        return integrall

    result = integrate.quad(lambda x: Integrand(x), -limit, limit)
    integrall = result[0]

    # compute Call price
    price = 1*S-K*np.exp(-r*tau)*integrall / (2*np.pi)

    return price

#Test (should give 4.1732)
print("S:     100.00")
print("K:     100.00")
print("tau:     1.0")
print("r:       0.01")
print("kappa:   1.98")
print("theta:   0.1**2")
print("sigma:   0.33")
print("rho:     0.025")
print("nu:      0.1197**2")
print("limit: 50")
price = heston_call(S=100, K=100, tau=1.0, r=.0, kappa=1.98, theta=0.108977**2, sigma=0.33, rho=0.025, nu=0.1197**2, limit=50)
print(f"Heston Call: {price:.4f}")
