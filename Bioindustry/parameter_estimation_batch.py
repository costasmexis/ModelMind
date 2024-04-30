import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import odeint, solve_ivp
from utils import get_data, get_training_data, solve_ode, plot_solution

from PINN import PINN, get_loss, get_loss_sparse

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print only 4 decimals
np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)


PATH = "./data/"
N_SAMPLES = 100 # If changes, update it in utils.py
EPOCHS = 30000
weight = 1.0

for exp_id in ["BR01", "BR02", "BR03", "BR04", "BR05", "BR06", "BR07", "BR08", "BR09"]:
    print(f"\nExperiment: {exp_id}")
    df = get_data(exp_id=exp_id, batch=True)
    t_train, u_train = get_training_data(df)

    # Train data to tensor
    ts_train = torch.tensor(t_train, requires_grad=True, device=device, dtype=torch.float32).view(-1, 1)
    us_train = torch.tensor(u_train, requires_grad=True, device=device, dtype=torch.float32)

    # Define the model
    pinn = PINN(1, 2, t_start=t_train.min(), t_end=t_train.max()).to(device)
    optimizer = torch.optim.RMSprop(pinn.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Train the model
    LOSS = []
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        u_pred = pinn(ts_train)
        residual_pred = get_loss(pinn)
        # residual_pred = get_loss_sparse(pinn)
        loss = weight * criterion(u_pred, us_train)
        loss += residual_pred
        loss.backward()
        optimizer.step()
        LOSS.append(loss.item())

        if epoch % 500 == 0:
            print(
            f"Epoch: {epoch} | Loss: {loss.item():.4f}, DATA Loss: {weight*criterion(u_pred, us_train):.4f}, ODE Loss: {torch.mean(residual_pred).item():.4f}"
            )
            print(
            f"mu_max: {pinn.mu_max.item():.4f}, Km: {pinn.Km.item():.4f}, Y_XS: {pinn.Y_XS.item():.4f}"
            )

        # Check if Y_XS > 1 and reset it 
        if pinn.Y_XS.item() > 1.0:
            pinn.Y_XS.data = torch.tensor([0.8], device=device, dtype=torch.float32)
        if pinn.Km.item() < 0.0:
            pinn.Km.data = torch.tensor([0.2], device=device, dtype=torch.float32)
        
        if len(LOSS) > 5000:
            if all(abs(loss_value - LOSS[-1]) < 0.10 * LOSS[-1] for loss_value in LOSS[-100:]) and loss.item() < 1.0:
                print(f"Early stopping at epoch {epoch}")
                break
            elif all([loss_value < 0.03 for loss_value in LOSS[-10:]]):
                print(f"Early stopping at epoch {epoch}")
                break

        
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
