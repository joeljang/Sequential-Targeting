import logging
import typing
from typing import Dict, List
from pathlib import Path

from torchtext.data import Field
from torch import nn
import torch

logging.basicConfig(level=logging.DEBUG,
                    filename='train.log',
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger = logging.getLogger()
logger.addHandler(console)

class RNNClassifier(nn.Module):
    def __init__(self, embedding, emb_drop, rnn, rnn_dim, bidir, rnn_drop):
        super().__init__()
        self.word_embeddings = embedding
        self.word_embeddings_drop = nn.Dropout(emb_drop)
        rnn = nn.LSTM if rnn == 'LSTM' else nn.GRU
        self.rnn = rnn(self.word_embeddings.weight.shape[1], rnn_dim, 2, bidirectional=bidir, batch_first=True)
        self.rnn_drop = nn.Dropout(rnn_drop)
        output_dim = rnn_dim * 4 if bidir else 2
        self.rnn_bn = nn.BatchNorm1d(output_dim)
        self.output = nn.Linear(output_dim, 1)

    @staticmethod
    def default_emb(vocab_size, emb_dim):
        emb = nn.Embedding(vocab_size, emb_dim)
        return emb

    def forward(self, sentences): # batch, sentence
        embeds = self.word_embeddings(sentences)
        embeds = self.word_embeddings_drop(embeds)
        rnn_out, _ = self.rnn(embeds)
        max_rnn_out, _ = rnn_out.max(0)
        avg_rnn_out = rnn_out.mean(0)
        cat_rnn = torch.cat((max_rnn_out, avg_rnn_out), -1)
        cat_rnn = self.rnn_drop(cat_rnn)

        return torch.sigmoid(self.output(cat_rnn))


    def save_as_torch_script(self, forward_fn, example, path):
        torch_f = torch.jit.trace(forward_fn, (example))
        torch_f.save(path)

    def save(self, model: nn.Module, fields: Dict[str, Field], path: Path):
        torch.save({'models': model, 'fields': fields}, str(path))