import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
import os

files = os.listdir('./data/')
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

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 2)
        )
        
        self.mu_max = nn.Parameter(torch.tensor([0.5]))
        self.K = nn.Parameter(torch.tensor([0.5]))
        self.Y = nn.Parameter(torch.tensor([0.5]))
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, x):
        return self.net(x)
    
def residuals(t, model: nn.Module):
    t_min = t.min().item()
    t_max = t.max().item()
    t = torch.linspace(t_min, t_max, 100).view(-1,1)
    t.requires_grad = True
    y = model(t)
    X = y[:, 0].view(-1,1)
    S = y[:, 1].view(-1,1)
    dX_dt = torch.autograd.grad(X, t, grad_outputs=torch.ones_like(X), create_graph=True)[0]
    dS_dt = torch.autograd.grad(S, t, grad_outputs=torch.ones_like(S), create_graph=True)[0]
    error_1 = dX_dt - model.mu_max * S * X / (model.K + S)
    error_2 = dS_dt + model.Y * model.mu_max * X * S/ (model.K + S)
    e_1 = torch.mean(error_1**2)
    e_2 = torch.mean(error_2**2)
    return e_1 + e_2 
    

def generate_traindata(df: pd.DataFrame):
    t_train = df['Time'].values.astype(np.float32)
    X_train = df['Biomass'].values
    S_train = df['Glucose'].values  
    y_train = np.hstack((X_train.reshape(-1, 1), S_train.reshape(-1, 1))).astype(np.float32)
    return t_train, y_train
    
                
############################################
df = get_data(files)
df = df[df['Batch'] == 0]
print(df.shape)

### Modeling ###
network = Network()
optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
criterion = nn.MSELoss()

EPOCHS = 10
LOSS = []
for epoch in range(EPOCHS):
    loss_exp = 0
    for exp in range(1, 10):
        _ = df[df['BR'] == exp]
        t_train, y_train = generate_traindata(_)
        ts_train = torch.tensor(t_train, requires_grad=True, dtype=torch.float32).view(-1,1)
        ys_train = torch.tensor(y_train, requires_grad=True, dtype=torch.float32)

        # Training
        optimizer.zero_grad()        
        pred_y = network.forward(ts_train)
        physics_loss = residuals(ts_train, network)
        
        loss = criterion(pred_y, ys_train)
        loss += physics_loss
        
        loss_exp += loss.item()
        loss.backward()
        optimizer.step()
        
    avg_loss = loss_exp / 9
    LOSS.append(avg_loss)
        
    if epoch % 100 == 0:
        print(f'Epoch: {epoch} | Loss: {avg_loss}, \n.......mu_max: {network.mu_max.item()}, K: {network.K.item()}, Y: {network.Y.item()}')
        
# Plot LOSS
plt.plot(LOSS)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.savefig('loss.png')
plt.close()
plt.plot(LOSS)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.savefig('loss.png')
plt.close()