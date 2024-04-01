import numpy as np
import torch
import torch.nn as nn
import seaborn as sns

T_START = 0
T_END = 4.1
N_SAMPLES = 100 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)

class PINN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PINN, self).__init__()
        self.input = nn.Linear(input_dim, 128)
        self.hidden = nn.Linear(128, 128)
        self.output = nn.Linear(128, output_dim)
        
        self.mu_max = nn.Parameter(torch.tensor([0.5]))
        self.Km = nn.Parameter(torch.tensor([0.5]))
        self.Y_XS = nn.Parameter(torch.tensor([0.5]))
    
    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.hidden(x))
        x = self.output(x)
        return x    
    
def get_loss(model: nn.Module):
    t = torch.linspace(T_START, T_END, N_SAMPLES, device=DEVICE).reshape(-1, 1)
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
