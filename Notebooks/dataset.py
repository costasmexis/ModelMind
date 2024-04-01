import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from scipy.integrate import odeint, solve_ivp

PATH_TO_DATA = '../data_ceasar.xlsx'
T_START = 0
T_END = 4.1
N_SAMPLES = 100 

class Dataset:
    def __init__(self):
        self.df = pd.read_excel(PATH_TO_DATA)
        self.t_start = T_START
        self.t_end = T_END
        self.n_samples = N_SAMPLES
        
        self.y0 = [self.df['Biomass'].iloc[0], self.df['Glucose'].iloc[0]] # (X, S)
        
        
    def get_data(self, exp: int = 1):
        return self.df[self.df['exp'] == exp]
    
    def __ode_func(t, y, mu_max, Km, Y_XS):
        X = y[0]
        S = y[1]
        mu = mu_max * S / (S + Km)
        dydt = [mu * X, -1/Y_XS * mu * X]
        return dydt        
    
    def solve_ode(self, mu_max, Km, Y_XS):
        t = np.linspace(self.t_start, self.t_end, self.n_samples)
        sol = odeint(Dataset.__ode_func, self.y0, t, args=(mu_max, Km, Y_XS))
        return sol