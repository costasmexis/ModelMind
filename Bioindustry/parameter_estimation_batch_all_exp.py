import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from utils import concat_data, get_training_data

from PINN import PINN, get_loss, get_loss_sparse

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = "./data/"
N_SAMPLES = 100
EPOCHS = 100000

# ******************** #
df = concat_data()

t_train, u_train = get_training_data(df)

# Train data to tensor
ts_train = torch.tensor(t_train, requires_grad=True, device=device, dtype=torch.float32).view(-1, 1)
us_train = torch.tensor(u_train, requires_grad=True, device=device, dtype=torch.float32)

def main():
    # Define the model
    pinn = PINN(1, 2, t_start=t_train.min(), t_end=t_train.max()).to(device)
    optimizer = torch.optim.RMSprop(pinn.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    
    # Train the model
    LOSS = []
    weight = 1.0
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
                f"Epoch: {epoch} | Loss: {loss.item()}, DATA Loss: {weight*criterion(u_pred, us_train)}, ODE Loss: {torch.mean(residual_pred).item()}"
            )
            print(
                f"mu_max: {pinn.mu_max.item()}, Km: {pinn.Km.item()}, Y_XS: {pinn.Y_XS.item()}"
            )
        
        # Check if Y_XS > 1 and reset it 
        if pinn.Y_XS.item() > 1.0:
            pinn.Y_XS.data = torch.tensor([0.8], device=device, dtype=torch.float32)
        if pinn.Km.item() < 0.0:
            pinn.Km.data = torch.tensor([0.2], device=device, dtype=torch.float32)

    # Plot the loss
    plt.plot(LOSS)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.savefig('./plots/loss_sparse.png')

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
    print(f'Using device: {device}')
    main()
    # main_batch_training()