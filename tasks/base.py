import json
import typing
from typing import Dict, List
from collections import Counter
from pydoc import locate

from tqdm import tqdm
from pyhdfs import HdfsClient
import torch
from torchtext.data import TabularDataset, Dataset, LabelField, Field, Example, Iterator
from torchtext.vocab import Vocab


class CorpusBase(object):
    C3_HDFS_HOST = 'c3.nn01.nhnsystem.com:50070'
    MAX_LEN = 256
    UNK_TOKEN = '<unk>'
    PAD_TOKEN = '<pad>'
    SPACE_TOKEN = '<sp>'
    INIT_TOKEN = '<s>'
    EOS_TOKEN = '<e>'
    TOKENS = [PAD_TOKEN, UNK_TOKEN, SPACE_TOKEN, INIT_TOKEN, EOS_TOKEN]
    FIELDS_TOKEN_ATTRS = ['init_token', 'eos_token', 'unk_token', 'pad_token']
    FIELDS_ATTRS = FIELDS_TOKEN_ATTRS + ['sequential', 'use_vocab',  'fix_length']

    def __init__(self, user_name, corpus_name):
        self.user_name = user_name
        self.corpus_name = corpus_name

    def extract_feature(self, example_dict:Dict[str, object]):
        raise NotImplementedError

    @property
    def corpus_path(self) -> str:
        return f'/corpus/{self.corpus_name}.jsons'

    @property
    def fields_path(self) -> str:
        return f'/corpus/{self.corpus_name}.fields'

    @property
    def c3_path(self) -> str:
        return f'/user/{self.user_name}/fortuna/pped/pped__{self.corpus_name}.jsons'

    @property
    def c3_fields_path(self) -> str:
        return f'/user/{self.user_name}/fortuna/pped/pped__{self.corpus_name}.fields'

    def __push_preprocessed(self, c3_path:str, user_name:str, dataset: Dataset):
        def push_to_hdfs(jstrs):
            if not fs.exists(c3_path):
                fs.create(c3_path, '\n'.join(jstrs) + '\n')
            else:
                fs.append(c3_path, '\n'.join(jstrs) + '\n')

        fs = HdfsClient(self.C3_HDFS_HOST, user_name=user_name)
        fs.mkdirs('/'.join(c3_path.split('/')[:-1]))
        fs.delete(c3_path)
        jstrs = []
        BUFSIZE = 2048
        for fxed_instance in tqdm(Iterator(dataset, batch_size=1), maxinterval=len(dataset.examples)):
            fxed_instance_dict = {name: getattr(fxed_instance, name).tolist()[0] for name in self.fields.keys()}
            jstrs.append(json.dumps(fxed_instance_dict))
            if len(jstrs) >= BUFSIZE:
                push_to_hdfs(jstrs)
                jstrs = []

        if jstrs:
            push_to_hdfs(jstrs)

    def __push_fields(self, hdfs_host: str, fields: Dict[str, Field]):
        fs = HdfsClient(hdfs_host)
        fs.mkdirs('/'.join(self.fields_path.split('/')[:-1]))
        fs.delete(self.fields_path)
        dicted_fields = {k: self.field_to_dict(v) for k, v in fields.items()}
        fs.create(self.fields_path, json.dumps(dicted_fields))

        fs = HdfsClient(self.C3_HDFS_HOST, user_name=self.user_name)
        fs.mkdirs('/'.join(self.c3_fields_path.split('/')[:-1]))
        fs.delete(self.c3_fields_path)
        c3_dicted_fields = {}
        for k, value in dicted_fields.items():
            if value['use_vocab']:
                max_vocab_index = len(value['vocab']['itos'])
                value['max_vocab_index'] = max_vocab_index
                value['dtype'] = str(torch.int64)
                vocab = value['vocab']
                for tok in self.FIELDS_TOKEN_ATTRS:
                    if value[tok]:
                        value[tok] = vocab['stoi'][value[tok]]
                value.pop('vocab')
                value['use_vocab'] = False
            else:
                value['max_vocab_index'] = 1
            c3_dicted_fields[k] = value
        fs.create(self.c3_fields_path, json.dumps(c3_dicted_fields))

    def load_fields_with_vocab(self, hdfs_host: str) -> Dict[str, Field]:
        fs = HdfsClient(hdfs_host)
        if fs.exists(self.fields_path):
            print(f'get fields from {hdfs_host}{self.fields_path}')
        else:
            raise Exception(f'there are no fields in {hdfs_host}{self.fields_path}')

        loaded_dict = json.loads(fs.open(self.fields_path).read())
        return {k: self.dict_to_field(v) for k, v in loaded_dict.items()}

    def field_to_dict(self, field: Field) -> Dict:
        dicted = {'type': f'{type(field).__module__}.{type(field).__name__}',
                  'dtype': str(field.dtype)}
        for k in self.FIELDS_ATTRS:
            dicted[k] = getattr(field, k)

        v = field.vocab
        vocab = {'itos': v.itos,
                 'stoi': v.stoi,
                 'unk_index': v.unk_index
                 }
        if hasattr(v, 'freqs'):
            vocab['freqs'] = dict(v.freqs)
        dicted['vocab'] = vocab
        return dicted

    def dict_to_field(self, dicted: Dict) -> Field:
        field = locate(dicted['type'])(dtype=locate(dicted['dtype']))
        for k in self.FIELDS_ATTRS:
            setattr(field, k, dicted[k])

        if 'vocab' in dicted:
            v_dict = dicted['vocab']
            vocab = Vocab(Counter())
            vocab.itos = v_dict['itos']
            vocab.stoi.update(v_dict['stoi'])
            vocab.unk_index = v_dict['unk_index']
            if 'freqs' in v_dict:
                vocab.freqs = Counter(v_dict['freqs'])
        else:
            vocab = Vocab(Counter())
            field.use_vocab = False
        field.vocab = vocab

        return field

    def __load_corpus_from_hdfs(self, hdfs_host: str) -> List:
        fs = HdfsClient(hdfs_host)
        with fs.open(self.corpus_path) as fp:
            corpus = list()
            for line in tqdm(fp.read().decode().split('\n')):
                if line:
                    d = json.loads(line)
                    corpus.append(d)
        return corpus

    def standby_corpus_to_c3(self, hdfs_host: str, fields: Field = None):
        print('corpus')
        self.corpus = self.__load_corpus_from_hdfs(hdfs_host)
        print('fxed_corpus')
        self.fxed_corpus = list(map(self.extract_features, self.corpus))

        print('dataset')
        if fields:
            self.fields = fields
            self.dataset = Dataset(self.fxed_corpus, fields=self.fields)
        else:
            self.dataset = Dataset(self.fxed_corpus, fields=self.fields)
            print('_build_vocabs')
            self._build_vocabs(self.fields, self.dataset)
            print('__push_fields')
            self.__push_fields(hdfs_host, self.fields)

        print('__push_preprocessed')
        self.__push_preprocessed(self.c3_path, self.user_name, self.dataset)
        print('done')

    def load_dataset(self, fields:Dict[str, Field]):
        preprocessed = self._load_preprocessed()
        return Dataset(preprocessed, fields)

    def _load_preprocessed(self) -> List[Example]:
        fs = HdfsClient(self.C3_HDFS_HOST, user_name=self.user_name)
        if fs.exists(self.c3_path):
            print(f'get preprocessed corpus from {self.C3_HDFS_HOST}{self.c3_path}')
        else:
            raise Exception(f'there are no preprocessed in {self.C3_HDFS_HOST}{self.c3_path}')

        preprocessed = []
        for line in fs.open(self.c3_path).read().decode().split('\n'):
            if line:
                ex = Example()
                for k, v in json.loads(line).items():
                    setattr(ex, k, v)
                preprocessed.append(ex)
        return preprocessed

    def load_fields_from_c3(self) -> Dict[str, Field]:
        fs = HdfsClient(self.C3_HDFS_HOST, user_name=self.user_name)
        if fs.exists(self.c3_fields_path):
            print(f'get fields from {self.C3_HDFS_HOST}{self.c3_fields_path}')
        else:
            raise Exception(f'there are no fields in {self.C3_HDFS_HOST}{self.c3_fields_path}')
        loaded_dict = json.loads(fs.open(self.c3_fields_path).read())
        print(loaded_dict)
        max_vocab_indexes = {k: v['max_vocab_index'] for k, v in loaded_dict.items()}
        return {k: self.dict_to_field(v) for k, v in loaded_dict.items()}, max_vocab_indexes


class Task(object):
    def __init__(self, user_name):
        self.user_name = user_name

    def load_dataset(self):
        raise NotImplementedError

    def build_dataset(self, hdfs_host):
        raise NotImplementedError
