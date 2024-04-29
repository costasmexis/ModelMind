import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

PATH = "./data/"

def ode_func(t, y, mu_max, Km, Y_XS):
    X = y[0]
    S = y[1]
    mu = mu_max * S / (S + Km)
    dydt = [mu * X, -1 / Y_XS * mu * X]
    return dydt


def solve_ode(mu_max, Km, Y_XS, t_start, t_end, y0, n_samples):
    t = np.linspace(t_start, t_end, n_samples)
    sol = solve_ivp(ode_func, [t_start, t_end], y0, t_eval=t, args=(mu_max, Km, Y_XS))
    return sol


def plot_solution(
    sol, mu_max, Km, Y_XS, df: pd.DataFrame, exp_id: str, 
    mae_biomass, mae_glucose, rmse_biomass, rmse_glucose, plot_exp: bool = True
):
    fig, ax = plt.subplots(figsize=(12, 8))
    if plot_exp:
        ax.scatter(df["Time"], df["Glucose"], label="Glucose", color="blue")
        ax.scatter(df["Time"], df["Biomass"], label="Biomass", color="green")
    ax.plot(sol.t, sol.y[0], label="X", color="red")
    ax.plot(sol.t, sol.y[1], label="S", color="orange")
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.set_title(
        f"mu_max: {mu_max:.2f}, Km: {Km:.2f}, Y_XS: {Y_XS:.2f}\n"
        f"MAE Biomass: {mae_biomass:.2f}, MAE Glucose: {mae_glucose:.2f}\n"
        f"RMSE Biomass: {rmse_biomass:.2f}, RMSE Glucose: {rmse_glucose:.2f}",
        fontsize=8
    )
    ax.legend()
    ax.grid(True)
    plt.savefig(f"./plots/solution_{exp_id}.png")
    plt.close()


def get_data(exp_id: str, batch: bool = True) -> pd.DataFrame:
    xls = pd.ExcelFile(PATH + f"{exp_id}_for_model.xlsx")
    df = xls.parse(0)
    df.drop(0, inplace=True)
    df['exp_id'] = exp_id
    if batch:
        df = df[df["Batch"] == 0]
    return df

def concat_data():
    return pd.concat([get_data(exp_id=exp_id, batch=True) for exp_id in ["BR01", "BR02", "BR03", "BR04", "BR05", "BR06", "BR07", "BR08", "BR09"]], ignore_index=True)

def get_training_data(df: pd.DataFrame) -> pd.DataFrame:
    t_train = df["Time"].values
    u_train = df[["Biomass", "Glucose"]].values
    return np.float32(t_train), np.float32(u_train)
