import json
import typing
from typing import Dict, List

from tqdm import tqdm
from pyhdfs import HdfsClient
from torchtext.data import TabularDataset, Dataset, LabelField, Field, Example, Iterator
import torch

from .base import CorpusBase, Task

class Tokenize(Task):
    def __init__(self, user_name: str, config):
        super().__init__(user_name)
        self.vocabfile = config['vocabfile']
        self.tokenizeword = config['tokenizeword']
        print('no problemo')
        self.files = config['files']
        self.vocab_corpus = self.Corpus(self.user_name, self.vocabfile, self.tokenizeword)
        corpus=[]
        for f in self.files:
            c = self.Corpus(self.user_name, f, self.tokenizeword)
            corpus.append(c)
        self.corpus = corpus

    def push_dataset(self, hdfs_host):
        print('build preprocessed')
        self.vocab_corpus.standby_corpus_to_c3(hdfs_host)
        for f in self.corpus:
            f.standby_corpus_to_c3(hdfs_host,self.vocab_corpus.fields)

    class Corpus(CorpusBase):
        def __init__(self, user_name: str, corpus_name: str, tokenizeword):
            super().__init__(user_name, corpus_name)
            self.tokenizeword = tokenizeword
            self.fields = self._build_fields()

        def _build_fields(self) -> Dict[str, Field]:
            fields = {'syllable_contents': Field(sequential=True, use_vocab=True, batch_first=True),
                      'label': LabelField(sequential=False, use_vocab=False, dtype=torch.float32, batch_first=True)}
            return fields

        def _build_vocabs(self, fields, dataset):
            fields['syllable_contents'].build_vocab(dataset, min_freq=5, max_size=None,
                                                    specials=self.TOKENS)
            fields['label'].build_vocab(dataset, max_size=None, specials=[])

        def extract_features(self, instance: Dict[str, object]) -> Example:
            try:
                wordslst = instance['review'].split()
                words = [self.SPACE_TOKEN if x == ' ' else x for x in wordslst[:self.MAX_LEN - 2]]
                syllables = [self.SPACE_TOKEN if x == ' ' else x for x in instance['review'][:self.MAX_LEN - 2]]
            except:
                print(instance)
            ex = Example()
            if(self.tokenizeword):
                setattr(ex, 'syllable_contents', [self.INIT_TOKEN] + words + [self.EOS_TOKEN])
            else:
                setattr(ex, 'syllable_contents', [self.INIT_TOKEN] + syllables + [self.EOS_TOKEN])
            if 'sentiment' in instance:
                label = instance['sentiment']
                if type(label) is int:
                    label = int(label)
                    setattr(ex, 'label', 1. if label >= 1 else 0.)
                elif type(label) is str:
                    '''
                    if(label=='NEG'):
                        setattr(ex, 'label', 0.)
                    elif(label=='POS'):
                        setattr(ex, 'label', 1.)
                    else:
                        setattr(ex, 'label', 2.)
                    '''
                    setattr(ex, 'label', 1. if (label == '1.0' or label == '1') else 0.)
                else:
                    raise Exception("yo label your y correctly...")
            return ex
