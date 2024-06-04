import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Evolution(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers=1):
        super(Evolution,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0,batch_first=True)
        # self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0, batch_first=True, bidirectional=True)
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=0, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        # out, _ = self.bilstm(x, (h0, c0))
        # out, _ = self.gru(x, h0)
        return out
