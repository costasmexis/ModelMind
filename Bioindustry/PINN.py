import torch
import torch.nn as nn

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SAMPLES = 100

class PINN(nn.Module):
    def __init__(self, input_dim, output_dim, t_start=0, t_end=10):
        super(PINN, self).__init__()
        self.input = nn.Linear(input_dim, 32)
        self.hidden = nn.Linear(32, 32)
        self.output = nn.Linear(32, output_dim)

        # Initialize Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        self.mu_max = nn.Parameter(torch.tensor([0.5]))
        self.Km = nn.Parameter(torch.tensor([0.5]))
        self.Y_XS = nn.Parameter(torch.tensor([0.5]))

        self.t_start = t_start
        self.t_end = t_end

    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.hidden(x))  # Hidden layer
        x = self.output(x)
        return x


def get_loss(model: nn.Module):
    t = torch.linspace(model.t_start, model.t_end, N_SAMPLES, device=DEVICE).reshape(-1, 1)
    t.requires_grad = True
    u = model(t).to(DEVICE)
    u_X = u[:, 0].view(-1, 1)
    u_S = u[:, 1].view(-1, 1)
    
    u_t_X = torch.autograd.grad(
        u_X, t, grad_outputs=torch.ones_like(u_X), create_graph=True
    )[0]
    u_t_S = torch.autograd.grad(
        u_S, t, grad_outputs=torch.ones_like(u_S), create_graph=True
    )[0]

    error_1 = u_t_X - model.mu_max * u_X * u_S / (u_S + model.Km)
    error_2 = u_t_S + 1 / model.Y_XS * model.mu_max * u_X * u_S / (u_S + model.Km)
    
    e_1 = torch.mean(error_1**2)
    e_2 = torch.mean(error_2**2)
    
    return e_1 + e_2

def get_loss_sparse(model: nn.Module):
    t_0 = model.t_start
    t_1 = 2.0
    step = 0.3
    E_1 = 0
    E_2 = 0
    
    n_iter = 8
    for i in range(n_iter):
        t = torch.linspace(t_0, t_1, N_SAMPLES, device=DEVICE).reshape(-1, 1)
        t.requires_grad = True
        u = model(t).to(DEVICE)
        u_X = u[:, 0].view(-1, 1)
        u_S = u[:, 1].view(-1, 1)
        u_t_X = torch.autograd.grad(
            u_X, t, grad_outputs=torch.ones_like(u_X), create_graph=True
        )[0]
        u_t_S = torch.autograd.grad(
            u_S, t, grad_outputs=torch.ones_like(u_S), create_graph=True
        )[0]
        error_1 = u_t_X - model.mu_max * u_X * u_S / (u_S + model.Km)
        error_2 = u_t_S + 1 / model.Y_XS * model.mu_max * u_X * u_S / (u_S + model.Km)
        E_1 += torch.mean(error_1**2)
        E_2 += torch.mean(error_2**2)
        t_0 += step
        t_1 += step
        if t_1 > model.t_end:
            t_1 = model.t_end 
    return E_1 + E_2