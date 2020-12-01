from torch import nn
import torch

class CNN_Multi_Channel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, pre_trained_embedding=None, multi_channel=False):
        super(CNN_Multi_Channel, self).__init__()
        self.embedding_dim = embedding_dim
        self.pre_trained_embedding = pre_trained_embedding
        self.vocab_size = vocab_size
        self.multi_channel = multi_channel

        if pre_trained_embedding is None:
            self.embedding1 = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        else:
            self.embedding1 = nn.Embedding.from_pretrained(self.pre_trained_embedding, freeze=False, padding_idx=0)
            self.embedding2 = nn.Embedding.from_pretrained(self.pre_trained_embedding, freeze=True, padding_idx=0)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.conv1d3 = nn.Conv1d(self.embedding_dim, 64, 3)
        self.conv1d4 = nn.Conv1d(self.embedding_dim, 128, 4)
        self.conv1d5 = nn.Conv1d(self.embedding_dim, 256, 5)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)

        self.fc = nn.Linear(896, 3)
        self.fc1 = nn.Linear(896 * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3)
        
        self.batch_norm = nn.BatchNorm1d(896)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(128)

        nn.init.xavier_uniform(self.conv1d3.weight)
        nn.init.xavier_uniform(self.conv1d4.weight)
        nn.init.xavier_uniform(self.conv1d5.weight)
        nn.init.xavier_uniform(self.fc.weight)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc3.weight)



    def forward(self, x): 
        if self.multi_channel is False:
            x = self.embedding1(x).transpose(0, 1).transpose(1, 2)   # N E L
            x = self.dropout1(x)

            c3 = self.conv1d3(x)                                    # N H L-2
            c3 = self.tanh(c3)
            _c3 = c3
            c3, _ = torch.max(c3, 2)                                # N H
            _c3 = torch.mean(_c3, 2)
            
            c4 = self.conv1d4(x)                                    # N H L-3 
            c4 = self.tanh(c4)
            _c4 = c4
            c4, _ = torch.max(c4, 2)
            _c4 = torch.mean(_c4, 2)

            c5 = self.conv1d5(x)                                    # N H L-4 
            c5 = self.tanh(c5)
            _c5 = c5
            c5, _ = torch.max(c5, 2)
            _c5 = torch.mean(_c5, 2)
            
            x = torch.cat([c3, c4, c5], 1)                          # N 3H
            _x = torch.cat([_c3, _c4, _c5], 1)                      # N 3H
            x = torch.cat([x, _x], 1)                               # N 6H

    
            x = self.dropout2(x)  
            x = self.fc(x)                                         # N 3
            return x
        else:
            x1 = self.embedding1(x).transpose(0, 1).transpose(1, 2) 
            x1 = self.dropout1(x1)

            c3 = self.conv1d3(x1)                                    
            c3 = self.tanh(c3)
            _c3 = c3
            c3, _ = torch.max(c3, 2)                            
            _c3 = torch.mean(_c3, 2)
            
            c4 = self.conv1d4(x1)                                  
            c4 = self.tanh(c4)
            _c4 = c4
            c4, _ = torch.max(c4, 2)
            _c4 = torch.mean(_c4, 2)

            c5 = self.conv1d5(x1)                                  
            c5 = self.tanh(c5)
            _c5 = c5
            c5, _ = torch.max(c5, 2)
            _c5 = torch.mean(_c5, 2)
            
            x1 = torch.cat([c3, c4, c5], 1)                     
            _x1 = torch.cat([_c3, _c4, _c5], 1)                  
            x1 = torch.cat([x1, _x1], 1)


            x2 = self.embedding2(x).transpose(0, 1).transpose(1, 2) 
            x2 = self.dropout1(x2)

            c3 = self.conv1d3(x2)                                    
            c3 = self.tanh(c3)
            _c3 = c3
            c3, _ = torch.max(c3, 2)                            
            _c3 = torch.mean(_c3, 2)
            
            c4 = self.conv1d4(x2)                                  
            c4 = self.tanh(c4)
            _c4 = c4
            c4, _ = torch.max(c4, 2)
            _c4 = torch.mean(_c4, 2)

            c5 = self.conv1d5(x2)                                  
            c5 = self.tanh(c5)
            _c5 = c5
            c5, _ = torch.max(c5, 2)
            _c5 = torch.mean(_c5, 2)
            
            x2 = torch.cat([c3, c4, c5], 1)                     
            _x2 = torch.cat([_c3, _c4, _c5], 1)                  
            x2 = torch.cat([x2, _x2], 1)                               

            x = torch.cat([x1, x2], 1)                                  
            x = self.fc1(x)  
            x = self.tanh(x)
            # x = self.batch_norm1(x)
            x = self.dropout2(x)
            x = self.fc2(x)     
            x = self.tanh(x)
            x = self.batch_norm2(x) 
            x = self.fc3(x)                        
            return x