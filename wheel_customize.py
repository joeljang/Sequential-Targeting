import sys
import json
import math
import typing
from typing import Dict, List

from torch import nn, optim
import torch
from torchtext.data import Iterator
from tqdm import tqdm

from models.cnn_rnn_pooling import CNNRNNBinary
from models.cnn_rnn_pooling_multi import CNNRNNMulti

from tasks.binary import Binary
from tasks.multi import Multi 
from tasks.tokenize import Tokenize

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize

from pyhdfs import HdfsClient
import pickle
import numpy as np

import wandb
wandb.login(key='6525130c8b35bcd27b1bc36f79ad88847e7dd982')

class Wheel(object):

    MODEL = 'CNNRNNPooling'
    TASK = 'SB'

    def __init__(self,config):
        #Basic Configs
        self.device = 'cuda'
        self.__test_iter = None
        self.C3_HDFS_HOST = 'c3.nn01.nhnsystem.com:50070'

        #Config File
        self.username = config['username']
        self.batch_size = config['batch']
        self.max_epoch = config['epoch']
        self.threshold = config['threshold']
        self.testnum = config['testnum']
        self.loadmodel = config['loadmodel']
        self.num = config['modelnum']
        self.e = config['modelepo']
        self.pretrain = config['pretrain']
        self.vocabfile = config['vocabfile']
        self.trainfile = config['trainfile']
        self.testfile = config['testfile']
        self.projectname = config['projectname']
        self.labels = config['labels']
        self.hdfs_host = config['hdfs_host']
        self.pretrainfile = config['pretrain']
        #Loading Task
        if(config['tasktype']=='binary'):
            self.isbinary = True
        elif(config['tasktype']=='multi'):
            self.isbinary = False
        else:
            raise Exception('Tasktype should be binary or multi! Other tasks are not yet supported')

        #Loading Task
        if(self.isbinary):
            self.task = Binary(self.vocabfile,self.trainfile,self.testfile,self.username, hdfs_host=self.hdfs_host)
        else:
            self.task = Multi(self.vocabfile,self.trainfile,self.testfile,self.username, hdfs_host=self.hdfs_host)
        self.task.load_dataset()

        #Loadig Model
        if(self.isbinary):
            self.model = CNNRNNBinary(128, 3, 0.2, self.task.max_vocab_indexes['syllable_contents'], 128)
            self.loss_fn = nn.MSELoss()
        else:
            self.model = CNNRNNMulti(128, 3, 0.2, self.task.max_vocab_indexes['syllable_contents'], 128)
            self.loss_fn = nn.CrossEntropyLoss()
        print(self.model)
        self.model.to(self.device)

        #Recording log in WANDB Library
        wandb.init(project=self.projectname, name = f'{self.trainfile}_{self.testnum}')
        wandb.log({'Total Epoch':self.max_epoch,'Batch':self.batch_size})
        if(self.isbinary):
            wandb.log({'Threshold':self.threshold})

        #Loading pretrained model if needed
        if(self.loadmodel):
            self.load_model(self.pretrainfile,self.num,self.e)

    def log_to_c3dl(self, msg:str):
        print(msg, file=sys.stderr)

    @property
    def test_iter(self) -> Iterator:
        if self.__test_iter:
            self.__test_iter.init_epoch()
            return self.__test_iter
        else:
            self.__test_iter = Iterator(self.task.te_dataset, batch_size=self.batch_size, repeat=False,
                                        sort_key=lambda x: len(x.syllable_contents), train=False,
                                        device=self.device)
            return self.__test_iter

    def getscore(self,pred,truth):
        threshold= self.threshold
        pred_m=[]
        if(self.isbinary):
            average = 'binary'
            for p in pred:
                if(p>threshold):
                    pred_m.append(1)
                else:
                    pred_m.append(0)
        else:
            average = 'macro'
            for p in pred:
                i = p.index(max(p))
                pred_m.append(i)
        truth = np.array(truth)
        pred_m = np.array(pred_m)
        precision, recall, f1score, blah = precision_recall_fscore_support(truth ,pred_m, average=average)
        if(self.isbinary):
            rocauc = roc_auc_score(truth,pred_m)
            f1 = []
        else:
            f1 = f1_score(truth,pred_m, average=None)
            y = label_binarize(truth,classes=self.labels)
            rocauc = roc_auc_score(y,pred,multi_class='ovo')
        return precision, recall, f1score, f1, rocauc

    def train(self,savemodel):
        max_epoch = self.max_epoch
        optimizer = optim.Adam(self.model.parameters())
        total_len = len(self.task.tr_dataset)
        ds_iter = Iterator(self.task.tr_dataset, batch_size=self.batch_size, repeat=False,
                           sort_key=lambda x: len(x.syllable_contents), train=True, device=self.device)
        min_iters = 10
        for epoch in range(max_epoch):
            loss_sum, acc_sum, len_batch_sum = 0., 0., 0.
            ds_iter.init_epoch()
            tr_total = math.ceil(total_len / self.batch_size)
            tq_iter = tqdm(enumerate(ds_iter), total=tr_total, miniters=min_iters, unit_scale=self.batch_size,
                           bar_format='{n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}] {desc}')

            self.model.train()
            for i, batch in tq_iter:
                self.model.zero_grad()
                pred = self.model(batch.syllable_contents)
                label = torch.tensor(batch.label, dtype=torch.long, device=self.device)
                if(self.isbinary):
                    acc = torch.sum((torch.reshape(pred, [-1]) > 0.5) == (batch.label > 0.5), dtype=torch.float32)
                    loss = self.loss_fn(pred, batch.label)
                else:
                    acc = torch.sum((torch.argmax(pred, dim=1)) == label, dtype=torch.float32)
                    loss = self.loss_fn(pred, label)
                loss.backward()
                optimizer.step()
                len_batch = len(batch)
                len_batch_sum += len_batch
                acc_sum += acc.tolist()
                loss_sum += loss.tolist() * len_batch
                if i % min_iters == 0:
                    tq_iter.set_description(f'{epoch:2} loss: {loss_sum / len_batch_sum:.5}, acc: {acc_sum / len_batch_sum:.5}', True)

            tq_iter.set_description(f'{epoch:2} loss: {loss_sum / total_len:.5}, acc: {acc_sum / total_len:.5}', True)

            self.log_to_c3dl(json.dumps(
                {'type': 'train', 'dataset': self.task.tr_corpus.corpus_name,
                 'epoch': epoch, 'loss': loss_sum / total_len, 'acc': acc_sum / total_len}))
            trainloss = loss_sum/total_len
            acc_lst, total, prec,recall,f1score,f1s,rocauc = self.eval(self.test_iter, len(self.task.te_dataset),trainloss=trainloss, epoch=epoch)
            self.save_model(savemodel,self.model, f'e{epoch}')

    def eval(self, iter:Iterator, total:int, trainloss:float=0, epoch:int=0):
        tq_iter = tqdm(enumerate(iter), total=math.ceil(total / self.batch_size),
                       unit_scale=self.batch_size, bar_format='{r_bar}')
        pred_lst = list()
        truth_lst = list()
        acc_lst = list()
        label_lst = list()
        self.model.eval()
        for i, batch in tq_iter:
            preds = self.model(batch.syllable_contents)
            label = torch.tensor(batch.label, dtype=torch.long, device=self.device)
            if(self.isbinary):
                accs = torch.eq(preds > 0.5, batch.label > 0.5).to(torch.float)
            else:
                accs = torch.eq(torch.argmax(preds, dim=1), label).to(torch.long)
            label_lst += label.tolist()
            acc_lst += accs.tolist()
            pred_lst += preds.tolist()

        prec,recall,f1score,f1s,rocauc = self.getscore(pred_lst,label_lst)

        accuracy = sum(acc_lst)/total
        self.log_to_c3dl(json.dumps(
        {'type': 'test', 'dataset': self.task.te_corpus.corpus_name,
        'epoch': epoch, 'accuracy': accuracy, 'precision': prec, 'recall': recall, 'f1score': f1score, 'ROC-AUC':rocauc}))
    
        wandb.log({'Epoch':epoch,'Accuracy':accuracy,'Precision':prec,'Recall':recall,'F1Score':f1score, 'Trainloss':trainloss, 'ROC-AUC':rocauc})
        if(len(f1s)!=0):
            for i in range(len(f1s)):
                wandb.log({f'Class {i} F1Score':f1s[i]})

        return acc_lst, total, prec, recall, f1score, f1s, rocauc
    
    def save_model(self, savemodel, model, appendix=None):
        if(savemodel):
            c3_path = f'/user/{self.username}/fortuna/model/{self.trainfile}_{self.testnum}/model'
            fs = HdfsClient(self.C3_HDFS_HOST, user_name=self.username)
            if appendix:
                c3_path += f'_{appendix}'

            model_pickle = pickle.dumps(model.state_dict())
            try:
                fs.create(c3_path, model_pickle, overwrite=True)
            except Exception as e:
                print(e)
        else:
            file_name = f'data_out/model'
            if appendix:
                file_name += f'_{appendix}'
            torch.save({'model': model.state_dict(), 'task': type(self.task).__name__}, file_name)
    
    def load_model(self, train_dir, modelnum, appendix):
        print('~' * 100)
        c3_path = f'/user/{self.username}/fortuna/model/{train_dir}_{modelnum}/model_e{appendix}'
        print(c3_path)
        fs = HdfsClient(self.C3_HDFS_HOST, user_name=self.username)
        model_pickle = fs.open(c3_path)
        model_dict = pickle.load(model_pickle)
        self.model.load_state_dict(model_dict)
        acc_lst, total, prec,recall,f1score,f1s,rocauc = self.eval(self.test_iter, len(self.task.te_dataset))
        print('~' * 100)

    def evaluate(self, config):
        models = config['evalmodels']
        for m in models:
            self.load_model(m['model'],m['num'],m['epoch'])

if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)
    if(config['tokenize']):
        task = Tokenize(config['username'], config)
        task.push_dataset(config['hdfs_host'])
    else:
        wheel = Wheel(config)
        if(config['onlyeval']):
            wheel.evaluate(config)
        else:
            wheel.train(config['savemodel'])