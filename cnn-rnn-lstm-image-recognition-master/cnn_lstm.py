from torch import dropout
import torch.nn as nn
import torch.nn.functional as f
from cnn import CNN

from cnn import model


class CnnLstm(nn.Module):
    def __init__(self):
        super(CnnLstm, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(
            input_size=1568,
            hidden_size=64,
            num_layers=1,
            batch_first=True)
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.size()
        c_in = x.view(batch_size * time_steps, channels, height, width)
        _, c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, time_steps, -1)
        r_out, (_, _) = self.rnn(r_in)
        r_out2 = self.linear(r_out[:, -1, :])
        return f.log_softmax(r_out2, dim=1)

import torch

num_classes = 4
class AlexLstm(nn.Module):
    def __init__(self):
        super(AlexLstm, self).__init__()
        self.cnn = CNN()
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
        #model.classifier._modules['6'] = nn.Linear(4096, 4)
        #model.eval()
        #self.cnn = model()
        self.rnn = nn.LSTM(
            input_size=4096,
            hidden_size=256,
            num_layers=4,
            dropout =0.5,
            batch_first=True)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.size()
        c_in = x.view(batch_size * time_steps, channels, height, width)
        _, c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, time_steps, -1)
        r_out, (_, _) = self.rnn(r_in)
        r_out2 = self.linear(r_out[:, -1, :])
        return f.log_softmax(r_out2, dim=1)

        

