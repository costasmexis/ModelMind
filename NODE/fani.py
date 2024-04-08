import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint, odeint_adjoint
import torch
import torch.nn as nn
import scipy.integrate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Lotka - Volterra (LV) system of ODE
def lotka_volterra(t, y, alpha=3., beta=0.6, delta=0.5, gamma=4.):
    x, y = y
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return torch.tensor([dxdt, dydt])

# Solve using torchdiffeq odeint
t = torch.arange(0, 4, 0.08)
y_0 = torch.tensor([10., 3.])
y_true = odeint(lotka_volterra, y_0, t, method='dopri5')

# Plot the solution
plt.plot(t, y_true[:, 0], label='Prey')
plt.plot(t, y_true[:, 1], label='Predator')
plt.legend()
plt.savefig('lotka_volterra.png')
plt.close()

## NODE
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 2)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, t, y):
        return self.net(y)
    
niters = 1500
func = ODEFunc().to(device)
optimizer = torch.optim.Adam(func.parameters(), lr=1e-3)
criterion = nn.MSELoss()

LOSS = []
for iter in range(niters+1):
    optimizer.zero_grad()
    pred_y = odeint(func=func, y0=y_0, t=t, rtol=1e-7, atol=1e-9, method='dopri5')
    loss = criterion(pred_y, y_true)
    LOSS.append(loss.item())
    loss.backward()
    optimizer.step()
    
    if iter % 100 == 0:
        print('Iter {:04d} | Total Loss {:.6f}'.format(iter, loss.item()))


# Plot
plt.plot(LOSS)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iteration')
plt.savefig('loss.png')
plt.close()

# Plot results
plt.plot(t.numpy(), y_true.numpy(), 'r')
plt.plot(t.numpy(), pred_y.detach().numpy(), 'b')
plt.xlabel('Time')
plt.ylabel('x,y')
plt.savefig('results.png')
plt.close()

# Plot derivative
dydt = func(t, y_true)
plt.plot(t.detach().numpy(), dydt.detach().numpy())
plt.xlabel('Time')
plt.ylabel('Derivative')
plt.title('Derivative vs Time')
plt.savefig('derivative.png')
plt.close()
