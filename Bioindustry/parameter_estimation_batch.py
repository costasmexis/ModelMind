import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import odeint, solve_ivp
from utils import get_data, get_training_data, solve_ode, plot_solution

from PINN import PINN, get_loss

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = "./data/"
N_SAMPLES = 100
EPOCHS = 20000

# for exp_id in ["BR01", "BR02", "BR03", "BR04", "BR05", "BR06", "BR07", "BR08", "BR09"]:
for exp_id in ["BR01"]:
    df = get_data(exp_id=exp_id, batch=True)
    t_train, u_train = get_training_data(df)

    # Train data to tensor
    ts_train = torch.tensor(t_train, requires_grad=True, device=device, dtype=torch.float32).view(-1, 1)
    us_train = torch.tensor(u_train, requires_grad=True, device=device, dtype=torch.float32)

    # Define the model
    pinn = PINN(1, 2, T_START=t_train.min(), T_END=t_train.max()).to(device)
    optimizer = torch.optim.RMSprop(pinn.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    # Train the model
    LOSS = []
    for epoch in range(EPOCHS):
        u_pred = pinn(ts_train)
        residual_pred = get_loss(pinn)
        loss = criterion(u_pred, us_train)
        loss += residual_pred
        LOSS.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss.item()}, ODE Loss: {torch.mean(residual_pred).item()}"
            )
            print(
                f"mu_max: {pinn.mu_max.item()}, Km: {pinn.Km.item()}, Y_XS: {pinn.Y_XS.item()}"
            )
            
    # Calculate the MAE and RMSE
    u_pred = pinn(ts_train)
    mae_biomass = torch.mean(torch.abs(u_pred[:, 0] - us_train[:, 0])).item()
    mae_glucose = torch.mean(torch.abs(u_pred[:, 1] - us_train[:, 1])).item()
    rmse_biomass = torch.sqrt(torch.mean((u_pred[:, 0] - us_train[:, 0]) ** 2)).item()
    rmse_glucose = torch.sqrt(torch.mean((u_pred[:, 1] - us_train[:, 1]) ** 2)).item()

    sol_pinn = solve_ode(
        mu_max=pinn.mu_max.item(),
        Km=pinn.Km.item(),
        Y_XS=pinn.Y_XS.item(),
        t_start=t_train.min(),
        t_end=t_train.max(),
        y0=[df["Biomass"].iloc[0], df["Glucose"].iloc[0]],
        n_samples=N_SAMPLES,
    )
    
    plot_solution(
        sol_pinn,
        pinn.mu_max.item(),
        pinn.Km.item(),
        pinn.Y_XS.item(),
        df,
        exp_id,
        mae_biomass, mae_glucose, rmse_biomass, rmse_glucose,
        plot_exp=True,
    )
