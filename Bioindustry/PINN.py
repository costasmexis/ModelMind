import numpy as np
import torch
import torch.nn as nn
import seaborn as sns

N_SAMPLES = 100 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)

class PINN(nn.Module):
    def __init__(self, input_dim, output_dim, T_START=0, T_END=10):
        super(PINN, self).__init__()
        self.input = nn.Linear(input_dim, 32)
        # nn.init.normal_(self.input.weight)
        self.hidden = nn.Linear(32, 128)
        # nn.init.normal_(self.hidden.weight)
        self.output = nn.Linear(128, output_dim)
        # nn.init.normal_(self.output.weight)
        
        self.mu_max = nn.Parameter(torch.tensor([0.5]))
        self.Km = nn.Parameter(torch.tensor([0.5]))
        self.Y_XS = nn.Parameter(torch.tensor([0.5]))
        
        self.t_start = T_START
        self.t_end = T_END
    
    def forward(self, x):
        x = torch.tanh(self.input(x)) 
        x = torch.tanh(self.hidden(x)) # Hidden layer
        # x = torch.tanh(self.hidden(x)) # Hidden layer
        x = self.output(x)
        return x    
    
def get_loss(model: nn.Module):
    t = torch.linspace(model.t_start, model.t_end, N_SAMPLES, device=DEVICE).reshape(-1, 1)
    t.requires_grad = True
    u = model(t).to(DEVICE)
    u_X = u[:,0].view(-1,1)
    u_S = u[:,1].view(-1,1)
    u_t_X  = torch.autograd.grad(u_X, t, grad_outputs=torch.ones_like(u_X), create_graph=True)[0]
    u_t_S  = torch.autograd.grad(u_S, t, grad_outputs=torch.ones_like(u_S), create_graph=True)[0]
    error_1 = u_t_X - model.mu_max * u_X * u_S / (u_S + model.Km)
    error_2 = u_t_S + 1 / model.Y_XS * model.mu_max * u_X * u_S / (u_S + model.Km)
    e_1 = torch.mean(error_1**2)
    e_2 = torch.mean(error_2**2)
    return e_1 + e_2    


def train(model: nn.Module, EPOCHS: int, ts_train: torch.Tensor, us_train: torch.Tensor, criterion: nn.Module, optimizer: torch.optim.Optimizer):
    LOSS = []
    for epoch in range(EPOCHS):
        u_pred = model(ts_train)
        residual_pred = get_loss(model)
        loss = criterion(u_pred, us_train)
        loss += 0.5*residual_pred
        LOSS.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}, ODE Loss: {torch.mean(residual_pred).item()}')
            print(f'mu_max: {model.mu_max.item()}, Km: {model.Km.item()}, Y_XS: {model.Y_XS.item()}')
        
        if model.Y_XS.item() > 1.0:
            nn.init.uniform_(model.Y_XS)
        if model.K.item() < 0.0:
            nn.init.uniform_(model.K)

    return LOSS