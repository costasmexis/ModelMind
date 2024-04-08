import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchdiffeq import odeint, odeint_adjoint
import scipy.integrate
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = torch.Tensor([0.1], device='cpu')
y0 = torch.Tensor([200.0], device='cpu')
T0 = torch.Tensor([100.0], device='cpu')
t = torch.linspace(0, 60, 100, device='cpu')

def euler(func, t, dt, y):
  return dt * func(t, y)

class thermal_model(nn.Module):
    def forward(self, t, y):
        return -k*(y-T0)

# Generate training data
y_true = odeint(thermal_model(), y0, t, method='dopri5')

# NODE
class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        self.func = func

    def forward(self, y0, t, solver):
        solution = torch.empty(len(t), *y0.shape, dtype=y0.dtype, device=y0.device)
        solution[0] = y0
        
        j=1
        for t0, t1 in zip(t[:-1], t[1:])        :
            dy = solver(self.func, t0, t1-t0, y0)
            y1 = y0 + dy
            solution[j] = y1
            j += 1
            y0 = y1
        return solution
    

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
                
    def forward(self, t, y):
        return self.net(y)
    
# Training
niters = 3000

func = ODEFunc().to(device)
optimizer = torch.optim.RMSprop(func.parameters(), lr=1e-3)

LOSS = []
for iter in range(niters+1):
    optimizer.zero_grad()
    pred_y = odeint(func=func, y0=y0, t=t, rtol=1e-7, atol=1e-9, method='dopri5')
    loss = torch.mean(torch.abs(pred_y - y_true))
    LOSS.append(loss.item())
    loss.backward()
    optimizer.step()
    
    if iter % 100 == 0:
        print('Iter {:04d} | Total Loss {:.6f}'.format(iter, loss.item()))

# Plot LOSS
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
plt.ylabel('Temperature')
plt.title('Temperature vs Time')
plt.legend(['True', 'Predicted'])
plt.savefig('results.png')
plt.close()

# Plot derivatives
dydt_true = -k*(y_true-T0)
dydt_pred = func(t, y_true)
plt.plot(t.numpy(), dydt_true.numpy(), 'r')
plt.plot(t.numpy(), dydt_pred.detach().numpy(), 'b')
plt.xlabel('Time')
plt.ylabel('Derivative')
plt.title('Derivative vs Time')
plt.legend(['True', 'Predicted'])
plt.savefig('derivatives.png')
plt.close()