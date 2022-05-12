import torch
import torch.nn as nn
import torch.nn.functional as F

# 22 different input sets?
INPUT_LEN = 22
# Features from CNN
INPUT_DIM = 4096
# 4 classes
OUTPUT_LEN = 4
# Image size
HEIGHT = 320
WIDTH = 320


# Notes from the paper:
# sgd = SGD(lr=0.01, decay=0.005, momentum=0.9
# optimizer = CrossEntropyLoss

class Plant_LSTM(nn.Module):
    def __init__(self, input_size=4096, hidden_size=256, seq_len=22, num_layers=4, 
            batch_size=32, dropout = 0.5, device):
        super(Plant_LSTM, self).__init__()
        # Store params
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.device = device

        # Build model
        self.lstm1 = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            batch_first = True,
            dropout = self.dropout,
            num_layers = self.num_layers)
        )
        
        self.fc = nn.Linear(self.hidden_dims, n_predictions)
        # This depends on our loss function
        # Softmax is already baked into Crossentropy loss for example.
        # Since the original paper uses crossentropy loss, we probably don't need this
        # self.smax = nn.Softmax()
    )


    def forward(self, x, hidden=None):
       x, hidden = self.lstm1(x)
       x = x[:,-1,:]
       x = self.fc(x)
       return x, hidden



