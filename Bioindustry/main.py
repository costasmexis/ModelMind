import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint, solve_ivp

from dataset import Dataset
from PINN import PINN, get_loss

files = os.listdir('../Bioindustry/data/')
files = [f for f in files if f.startswith('BR')]
files.sort()

def get_data(files: list) -> pd.DataFrame:
    dfs = []
    for i, file in enumerate(files):
        df = pd.read_excel(f'./data/{file}', header=0)
        df = df[1:]
        df['BR'] = i+1
        dfs.append(df)
    return pd.concat(dfs)

df = get_data(files)
df = df[df['Batch'] == 0]
df.to_csv('./data_ceasar.csv')

d = Dataset()
d.get_data(exp=2)

pinn = PINN(1, 2, T_START=d.df['Time'].min(), T_END=d.df['Time'].max())
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-4)
criterion = nn.MSELoss()

t_train = d.df['Time'].values
u_train = d.df[['Biomass', 'Glucose']].values

# Train data to tensor
ts_train = torch.tensor(t_train, requires_grad=True).float().view(-1,1)
us_train = torch.tensor(u_train, requires_grad=True).float()

EPOCHS = 20000
LOSS = []
for epoch in range(EPOCHS):
    u_pred = pinn(ts_train)
    residual_pred = get_loss(pinn)
    loss = criterion(u_pred, us_train)
    loss += 0.5*residual_pred
    LOSS.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}, ODE Loss: {torch.mean(residual_pred).item()}')
        print(f'mu_max: {pinn.mu_max.item()}, Km: {pinn.Km.item()}, Y_XS: {pinn.Y_XS.item()}')
        
sol_pinn = d.solve_ode(mu_max=pinn.mu_max.item(), Km=pinn.Km.item(), Y_XS=pinn.Y_XS.item())
d.plot_solution(sol_pinn, 1)

