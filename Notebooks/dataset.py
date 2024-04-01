import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm 

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
        self.df = self.df[self.df['exp'] == exp]
        self.y0 = [self.df['Biomass'].iloc[0], self.df['Glucose'].iloc[0]] # (X, S)
    
    def __ode_func(self, t, y, mu_max, Km, Y_XS):
        X = y[0]
        S = y[1]
        mu = mu_max * S / (S + Km)
        dydt = [mu * X, -1/Y_XS * mu * X]
        return dydt        
    
    def solve_ode(self, mu_max, Km, Y_XS):
        t = np.linspace(self.t_start, self.t_end, self.n_samples)
        sol = solve_ivp(self.__ode_func, [self.t_start, self.t_end], self.y0, t_eval=t, args=(mu_max, Km, Y_XS))
        return sol
    
    def plot_solution(self, sol, plot_exp: bool = True):
        fig, ax = plt.subplots(figsize=(6, 4))
        if plot_exp:
            ax.scatter(self.df['Time'], self.df['Glucose'], label='Glucose', color='blue')
            ax.scatter(self.df['Time'], self.df['Biomass'], label='Biomass', color='green')
        ax.plot(sol.t, sol.y[0], label='X', color='red')
        ax.plot(sol.t, sol.y[1], label='S', color='orange')
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration')
        ax.set_title('Concentration vs Time')
        ax.legend()
        ax.grid(True)
        plt.show()
        
    def training_dataset(self, N: int = 1000):
        columns = [f'X_{i}' for i in range(len(self.df))]  + [f'S_{i}' for i in range(len(self.df))] + ['mu_max', 'Km', 'Y_XS']
        train = pd.DataFrame(columns=columns)
        t_train = np.linspace(self.t_start, self.t_end, self.df.shape[0])
        for i in tqdm(range(N)):
            mu_max = np.random.uniform(0.1, 1)
            Km = np.random.uniform(0.0001, 1)
            Y_XS = np.random.uniform(0.1, 1) 
            sol = solve_ivp(self.__ode_func, [self.t_start, self.t_end], self.y0, t_eval=t_train, args=(mu_max, Km, Y_XS))
            y_train_X = sol.y[0]
            y_train_S = sol.y[1]
            train.loc[i] = np.concatenate([y_train_X, y_train_S, [mu_max, Km, Y_XS]])
        # drop rows with negative values 
        train = train[(train >= 0).all(1)]
        return train