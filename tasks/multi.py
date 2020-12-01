import json
import typing
from typing import Dict, List

from tqdm import tqdm
from pyhdfs import HdfsClient
from torchtext.data import TabularDataset, Dataset, LabelField, Field, Example, Iterator
import torch

from .base import CorpusBase, Task

class Multi(Task):
    def __init__(self, vocabfile, trainfile, testfile, user_name: str, hdfs_host: str = None, ):
        super().__init__(user_name)
        self.vocabfile = vocabfile
        self.trainfile = trainfile
        self.testfile = testfile
        self.vocab_corpus = self.Corpus(self.user_name, self.vocabfile)
        self.tr_corpus = self.Corpus(self.user_name, self.trainfile)
        self.te_corpus = self.Corpus(self.user_name, self.testfile)

    def load_dataset(self):
        print('load fields')
        self.fields, self.max_vocab_indexes = self.vocab_corpus.load_fields_from_c3()

        print('load dataset')
        self.tr_dataset = self.tr_corpus.load_dataset(self.fields)
        self.te_dataset = self.te_corpus.load_dataset(self.fields)

    def push_dataset(self, hdfs_host):
        print('build preprocessed')
        self.vocab_corpus.standby_corpus_to_c3(hdfs_host)
        self.tr_corpus.standby_corpus_to_c3(hdfs_host, self.vocab_corpus.fields)
        self.te_corpus.standby_corpus_to_c3(hdfs_host, self.vocab_corpus.fields)

    class Corpus(CorpusBase):
        def __init__(self, user_name: str, corpus_name: str = 'catcalling'):
            super().__init__(user_name, corpus_name)
            self.fields = self._build_fields()

        def _build_fields(self) -> Dict[str, Field]:
            fields = {'syllable_contents': Field(sequential=True, use_vocab=True, batch_first=True),
                      'label': LabelField(sequential=False, use_vocab=False, dtype=torch.float32, batch_first=True)}
            return fields

        def _build_vocabs(self, fields, dataset):
            fields['syllable_contents'].build_vocab(dataset, min_freq=1, max_size=None,
                                                    specials=self.TOKENS)
            fields['label'].build_vocab(dataset, max_size=None, specials=[])

        def extract_features(self, instance: Dict[str, object]) -> Example:
            try:
                syllables = [self.SPACE_TOKEN if x == ' ' else x for x in instance['contents'][:self.MAX_LEN - 2]]
            except:
                print(instance)
            ex = Example()
            setattr(ex, 'syllable_contents', [self.INIT_TOKEN] + syllables + [self.EOS_TOKEN])
            if 'label' in instance:
                label = int(instance['label'])
                if label == 2:
                    setattr(ex, 'label', 2.)
                elif label == 1:
                    setattr(ex, 'label', 1.)
                elif label == 0:
                    setattr(ex, 'label', 0.)
            return ex
