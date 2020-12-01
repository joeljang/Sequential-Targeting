from torch import nn
import torch

class ELMO(nn.Module):
    def __init__(self, cell_type, n_vocab, d_vocab, d_rnn):
        super().__init__()
        self.cell_type = cell_type
        self.n_vocab = n_vocab
        self.d_vocab = d_vocab
        self.d_rnn = d_rnn

        self._embedding = nn.Embedding(self.n_vocab, self.d_vocab)
        self._act_e_r1 = nn.ReLU()

        self._fw_rnns = [nn.LSTM(self.d_vocab, self.d_rnn), nn.LSTM(self.d_rnn, self.d_rnn)]
        for i, m in enumerate(self._fw_rnns):
            self.add_module(f'_fw_rnns_{i}', m)
        self._bw_rnns = [nn.LSTM(self.d_vocab, self.d_rnn), nn.LSTM(self.d_rnn, self.d_rnn)]
        for i, m in enumerate(self._bw_rnns):
            self.add_module(f'_bw_rnns_{i}', m)

        self._output_linear = nn.Linear(self.d_rnn, self.n_vocab)


    def forward(self, x):
        '''LB'''
        x_len = torch.sum(x != 1, 0)
        emb = self._embedding(x)
        for i, (fw, bw) in enumerate(zip(self._fw_rnns, self._bw_rnns)):
            if i == 0:
                fw_state, _ = fw(emb)
                fw_out = torch.zeros_like(fw_state)
                bemb = self.flipBatch(emb.clone(), x_len)
                bw_state, _ = bw(bemb)
                bw_out = torch.zeros_like(fw_state)
            else:
                fw_state, _ = fw(fw_state)
                bw_state, _ = bw(bw_state)
            fw_out += fw_state
            bw_out += bw_state

        bw_out = self.flipBatch(bw_out, x_len)

        bw_out = torch.cat([bw_out[1:,:,:], torch.zeros_like(bw_out[0:1,:,:])], 0)
        fw_out = torch.cat([torch.zeros_like(fw_out[0:1,:,:]), fw_out[:-1,:,:]], 0)

        output = self._output_linear(fw_out + bw_out)
        return output

    def flipBatch(self, data, lengths):
        '''LBD'''
        assert data.shape[1] == len(lengths), "Batchsize Mismatch!"
        for i in range(data.shape[1]):
            data[:lengths[i], i, :] = data[:lengths[i], i, :].flip(dims=[0])
        return data
