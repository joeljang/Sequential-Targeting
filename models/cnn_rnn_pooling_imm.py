from torch import nn
import torch

class CNNRNNBinary(nn.Module):
    def __init__(self, hidden_dim, filter_size, dropout_rate, vocab_size, embedding_dim, pre_trained_embedding=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.filter_size = filter_size
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim

        if pre_trained_embedding is None:
            self.vocab_size = vocab_size
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(pre_trained_embedding, freeze=False, padding_idx=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.conv1d = nn.Conv1d(self.embedding_dim, self.hidden_dim, self.filter_size)
        self.bi_rnn = nn.LSTM(self.hidden_dim, int(self.hidden_dim / 2), batch_first=False, bidirectional=True)
        self.uni_rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=False)
        self.max_pool = nn.AdaptiveAvgPool2d((1, self.hidden_dim))
        self.linear = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def drop(self, tensor):
        binomial = torch.distributions.binomial.Binomial(probs=1-self.dropout_rate)
        mask = binomial.sample(tensor.size()).to(device='cuda')
        mask2 = (mask.int() ^ 1).float()
        result = tensor * mask * (1.0 / (1-self.dropout_rate))
        return result, mask2
    
    def forward(self, x, pastmodel=None):
        if(pastmodel==None):
            print('YO leego with no past modelll')
            # LN
            x = self.embedding(x).transpose(0, 1).transpose(1, 2)
            # LND NLD NDL
            x = self.conv1d(x).transpose(1, 2).transpose(0, 1)
            x = self.relu(x)
            x = self.dropout(x)
            x_res = x
            x, _ = self.bi_rnn(x)
            x, _ = self.uni_rnn(x + x_res)
            x = self.dropout(x)
            x, _ = torch.max(x, 0)
            x = self.linear(x)
            x = self.sigmoid(x).squeeze()
            return x
        else:
            print('YO leego WITH past modelll')
            # LN
            x = self.embedding(x).transpose(0, 1).transpose(1, 2)
            # LND NLD NDL
            y1 = self.conv1d(x).transpose(1, 2).transpose(0, 1)
            y1, m = self.drop(y1)
            y2 = pastmodel.conv1d(x).transpose(1, 2).transpose(0, 1)
            x = y1 + (m * y2)
            x = self.relu(x)
            #BI RNN
            x_res = x
            y1, _ = self.bi_rnn(x)
            y1, m = self.drop(y1)
            y2, _ = pastmodel.bi_rnn(x)
            x = y1 + (m * y2)
            #UNI RNN
            y1, _ = self.uni_rnn(x + x_res)
            y1, m = self.drop(y1)
            y2, _ = pastmodel.uni_rnn(x + x_res)
            x = y1 + (m * y2)
            x, _ = torch.max(x, 0)
            x = self.linear(x)
            x = self.sigmoid(x).squeeze()
            return x