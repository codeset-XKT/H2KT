import torch
import torch.nn as nn

class Predict(nn.Module):
    def __init__(self, embed_dim, num_class):
        super(Predict, self).__init__()
        self.hidden = nn.Linear(2*embed_dim, embed_dim)
        self.predict = nn.Linear(embed_dim, num_class)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        y = self.hidden(x)
        y = torch.relu(y)
        y = self.predict(y)
        y = self.dropout(y)
        y = torch.sigmoid(y).squeeze(-1)
        return y