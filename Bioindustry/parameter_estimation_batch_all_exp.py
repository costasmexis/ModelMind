import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import odeint, solve_ivp

from PINN import PINN, get_loss, get_loss_sparse

from utils import concat_data, get_training_data

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = "./data/"
N_SAMPLES = 100
EPOCHS = 10000

# Read data
df = concat_data()
t_train, u_train = get_training_data(df)

# Train data to tensor
ts_train = torch.tensor(t_train, requires_grad=True, device=device, dtype=torch.float32).view(-1, 1)
us_train = torch.tensor(u_train, requires_grad=True, device=device, dtype=torch.float32)

def main():
    # Define the model
    pinn = PINN(1, 2, T_START=t_train.min(), T_END=t_train.max()).to(device)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    # Train the model
    LOSS = []
    for epoch in range(EPOCHS):
        u_pred = pinn(ts_train)
        # residual_pred = get_loss(pinn)
        residual_pred = get_loss_sparse(pinn)
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

        # Check if Y_XS > 1 and reset it 
        if pinn.Y_XS.item() > 1.0:
            pinn.Y_XS.data = torch.tensor([0.8], device=device, dtype=torch.float32)
        if pinn.Km.item() < 0.0:
            pinn.Km.data = torch.tensor([0.2], device=device, dtype=torch.float32)
        if pinn.mu_max.item() > 1.0:
            pinn.mu_max.data = torch.tensor([0.8], device=device, dtype=torch.float32)


    u_pred = pinn(ts_train)
    mae_biomass = torch.mean(torch.abs(u_pred[:, 0] - us_train[:, 0])).item()
    mae_glucose = torch.mean(torch.abs(u_pred[:, 1] - us_train[:, 1])).item()
    rmse_biomass = torch.sqrt(torch.mean((u_pred[:, 0] - us_train[:, 0])**2)).item()
    rmse_glucose = torch.sqrt(torch.mean((u_pred[:, 1] - us_train[:, 1])**2)).item()

    print(
        f"MAE Biomass: {mae_biomass:.2f}, MAE Glucose: {mae_glucose:.2f}\n"
        f"RMSE Biomass: {rmse_biomass:.2f}, RMSE Glucose: {rmse_glucose:.2f}"
    )

    print(
        f"mu_max: {pinn.mu_max.item()}, Km: {pinn.Km.item()}, Y_XS: {pinn.Y_XS.item()}"
    )

    # Save the model
    torch.save(pinn.state_dict(), "./models/pinn_sparse.pth")



if __name__ == "__main__":
    main()