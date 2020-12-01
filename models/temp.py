# LN
            x = self.embedding(x).transpose(0, 1).transpose(1, 2)
            # LND NLD NDL
            y1 = self.conv1d(x).transpose(1, 2).transpose(0, 1)
            y1, m = self.drop(y1)
            y2 = pastmodel.conv1d(x).transpose(1, 2).transpose(0, 1)
            x = y1 + (y2 * m)
            x = self.relu(x)
            #BI RNN
            x_res = x
            y1, _ = self.bi_rnn(x)
            y1, m = self.drop(y1)
            y2, _ = pastmodel.bi_rnn(x)
            x = y1 + (y2 * m)
            #UNI RNN
            y1, _ = self.uni_rnn(x + x_res)
            y1, m = self.drop(y1)
            y2, _ = pastmodel.uni_rnn(x + x_res)
            x = y1 + (y2 * m)
            x, _ = torch.max(x, 0)
            x = self.linear(x)
            x = self.sigmoid(x).squeeze()
            return x