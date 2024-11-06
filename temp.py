import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MultiHeadAttentionRegression(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim,   )
        self.position_encoding = nn.Parameter(torch.randn(1, input_dim, input_dim))
        
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(input_dim)
            for _ in range(num_layers)
        ])
        
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim)
        )
        
        self.final_layer_norm = nn.LayerNorm(input_dim)
        self.output_layer = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        print(x.shape, self.position_encoding.shape)
        x = x + self.position_encoding[:, :x.size(1), :]
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, input_dim)
        
        for attention, norm in zip(self.attention_layers, self.layer_norms):
            residual = x
            x, _ = attention(x, x, x)
            x = self.dropout(x)
            x = norm(x + residual)
            
            residual = x
            x = self.feedforward(x)
            x = self.dropout(x)
            x = norm(x + residual)
        
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, input_dim)
        x = self.final_layer_norm(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.output_layer(x)

# Example usage
def train_model(X_train, y_train, X_val, y_val, input_dim, num_heads, num_layers, batch_size, epochs, lr):
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = MultiHeadAttentionRegression(input_dim, num_heads, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# Generate some example data
np.random.seed(42)
X = np.random.rand(1000, 12)
y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 1000)

# Split the data
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Train the model
input_dim = X.shape[1]
num_heads = 4
num_layers = 2
batch_size = 32
epochs = 50
lr = 0.001

train_model(X_train, y_train, X_val, y_val, input_dim, num_heads, num_layers, batch_size, epochs, lr)