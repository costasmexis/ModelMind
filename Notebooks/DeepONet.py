import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gstools import SRF, Gaussian
from gstools.random import MasterRNG

from scipy.integrate import odeint, solve_ivp

seed = MasterRNG(42)

T_START = 0
T_END = 1
N_SAMPLES = 100
N_CURVES = 50

def mm(t, y, mu_max, Km, Y_XS):
    X = y[0]
    S = y[1]
    mu = mu_max * S / (Km + S)
    dydt = [mu * X, -1 / Y_XS * mu * X]
    return dydt

def contois(t, y, mu_max, A, Y_XS):
    X = y[0]
    S = y[1]
    mu = mu_max * S / (A*X + S)
    dydt = [mu * X, -1 / Y_XS * mu * X]
    return dydt

def substrate_inh(t, y, mu_max, Km, Ki, Y_XS):
    X = y[0]
    S = y[1]
    mu = mu_max * S / (Km + S + S**2/Ki)
    dydt = [mu * X, -1 / Y_XS * mu * X]
    return dydt

t = np.linspace(T_START, T_END, N_SAMPLES)

S = np.zeros((N_SAMPLES, N_CURVES, 2))
U = np.zeros((N_SAMPLES, N_CURVES))

for i in range(N_CURVES):