import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import odeint, solve_ivp

from PINN import PINN, get_loss

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = "./data/"
N_SAMPLES = 100
EPOCHS = 20000


def ode_func(t, y, mu_max, Km, Y_XS):
    X = y[0]
    S = y[1]
    mu = mu_max * S / (S + Km)
    dydt = [mu * X, -1 / Y_XS * mu * X]
    return dydt


def solve_ode(mu_max, Km, Y_XS, t_start, t_end, y0, n_samples):
    t = np.linspace(t_start, t_end, N_SAMPLES)
    sol = solve_ivp(ode_func, [t_start, t_end], y0, t_eval=t, args=(mu_max, Km, Y_XS))
    return sol


def plot_solution(
    sol, mu_max, Km, Y_XS, df: pd.DataFrame, exp_id: str, plot_exp: bool = True
):
    fig, ax = plt.subplots(figsize=(6, 4))
    if plot_exp:
        ax.scatter(df["Time"], df["Glucose"], label="Glucose", color="blue")
        ax.scatter(df["Time"], df["Biomass"], label="Biomass", color="green")
    ax.plot(sol.t, sol.y[0], label="X", color="red")
    ax.plot(sol.t, sol.y[1], label="S", color="orange")
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.set_title(
        f"Concentration vs Time\nmu_max: {mu_max:.6f}, Km: {Km:.6f}, Y_XS: {Y_XS:.6f}"
    )
    ax.legend()
    ax.grid(True)
    plt.savefig(f"./plots/solution_{exp_id}.png")
    plt.close()


def get_data(exp_id: str) -> pd.DataFrame:
    xls = pd.ExcelFile(PATH + f"{exp_id}_for_model.xlsx")
    df = xls.parse(0)
    df.drop(0, inplace=True)
    return df


def get_batch_phase(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Batch"] == 0]


def get_training_data(df: pd.DataFrame) -> pd.DataFrame:
    t_train = df["Time"].values
    u_train = df[["Biomass", "Glucose"]].values
    return np.float32(t_train), np.float32(u_train)


### Example run on BR01
for exp_id in ["BR01", "BR02", "BR03", "BR04", "BR05", "BR06", "BR07", "BR08", "BR09"]:
    df = get_data(exp_id)
    df = get_batch_phase(df)
    t_train, u_train = get_training_data(df)

    # Train data to tensor
    ts_train = torch.tensor(
        t_train, requires_grad=True, device=device, dtype=torch.float32
    ).view(-1, 1)
    us_train = torch.tensor(
        u_train, requires_grad=True, device=device, dtype=torch.float32
    )

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
        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss.item()}, ODE Loss: {torch.mean(residual_pred).item()}"
            )
            print(
                f"mu_max: {pinn.mu_max.item()}, Km: {pinn.Km.item()}, Y_XS: {pinn.Y_XS.item()}"
            )

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
        plot_exp=True,
    )
