import json
import typing
from typing import Dict, List

from tqdm import tqdm
from pyhdfs import HdfsClient
from torchtext.data import TabularDataset, Dataset, LabelField, Field, Example, Iterator
import torch

from .base import CorpusBase, Task


class LMBase(Task):
    def __init__(self, user_name: str, hdfs_host: str = None):
        super().__init__(user_name)
        self.tr_corpus = self.Corpus(self.user_name, 'catcalling_0719_te')
        self.te_corpus = self.Corpus(self.user_name, 'catcalling_0719_te')

    def load_dataset(self):
        print('load fields')
        self.fields, self.max_vocab_indexes = self.tr_corpus.load_fields_from_c3()

        print('load dataset')
        self.tr_dataset = self.tr_corpus.load_dataset(self.fields)
        self.te_dataset = self.te_corpus.load_dataset(self.fields)

    def push_dataset(self, hdfs_host):
        print('build preprocessed')
        self.tr_corpus.standby_corpus_to_c3(hdfs_host)
        self.te_corpus.standby_corpus_to_c3(hdfs_host, self.tr_corpus.fields)

    class Corpus(CorpusBase):
        def __init__(self, user_name:str, corpus_name:str):
            super().__init__(user_name, corpus_name)
            self.fields = {'syllable_contents': Field(sequential=True, use_vocab=True, batch_first=True)}

        def _build_fields(self) -> Dict[str, Field]:
            fields = {'syllable_contents': Field(sequential=True, use_vocab=True, batch_first=True)}
            return fields

        def _build_vocabs(self, fields, dataset):
            fields['syllable_contents'].build_vocab(dataset, min_freq=5, max_size=None,
                                                    specials=self.TOKENS)

        def extract_features(self, instance: Dict[str, object]) -> Example:
            syllables = [self.SPACE_TOKEN if x == ' ' else x for x in instance['contents'][:self.MAX_LEN - 2]]
            ex = Example()
            setattr(ex, 'syllable_contents', [self.INIT_TOKEN] + syllables + [self.EOS_TOKEN])
            return ex

    @staticmethod
    def acc(pred, repr):
        eqs = torch.eq(pred, repr).to(torch.float)
        valid = torch.sum(repr != 1 , dtype=torch.float) # pad_idx
        invalid = repr.shape[0] - valid
        return max(torch.sum(eqs) - invalid, 0.) / valid


