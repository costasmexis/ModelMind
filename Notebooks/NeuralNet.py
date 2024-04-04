import torch
import torch.nn as nn
import numpy as np
import pandas as pd

DEVIDE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)

def preprocess_data(train_df: pd.DataFrame):
    X = torch.tensor(train_df.drop(columns=['mu_max', 'Km', 'Y_XS']).values, dtype=torch.float32)
    y = torch.tensor(train_df[['mu_max', 'Km', 'Y_XS']].values, dtype=torch.float32)
    return X, y

class NN:
    def __init__(self, input_dim: int = 16, output_dim: int = 3):
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def forward(self, X):   
        return self.model(X)
    
    def train_val_split(self, X, y, val_size: float = 0.2):
        n_samples = X.shape[0]
        n_val = int(n_samples * val_size)
        indices = np.random.permutation(n_samples)
        train_indices, val_indices = indices[n_val:], indices[:n_val]
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        return X_train, y_train, X_val, y_val
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.train()
        y_pred = self.model(X_train)
        loss = self.criterion(y_pred, y_train)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if X_val is not None and y_val is not None:
            y_pred_val = self.model(X_val)
            loss_val = self.criterion(y_pred_val, y_val)
            return loss.item(), loss_val.item()
        return loss.item()        
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X)
        
