from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import json

import numpy as np
import torch
from collections import namedtuple
from tempfile import TemporaryDirectory
from pathlib import Path
from torch.utils.data import (DataLoader, RandomSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import MSELoss

from general_distill import PregeneratedDataset
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling import BertForPreTraining
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from torch.nn import functional as F
import copy
import math
class DottableDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
        self.allowDotting()
    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()

def log_model(model):
    size = 0
    for n, p in model.named_parameters():
        #print('n: {}'.format(n))
        #print('p: {}'.format(p.nelement()))
        size += p.nelement()
    print('Total parameters: {}'.format(size))
    return size

path = "models/bert-base-uncased"
data = Path("data/pregenerated_data")
config = {"rank_condim": 64, "rank_dim": 64, "rank_layer": 144, "vocab_size": 30522, "hidden_size" : 768, "requires_grad": True, "num_hidden_layers": 12, "ops": "ffn","is_expand": False}
# config = {"rank_condim": 768, "rank_dim": 768, "rank_layer": 144, "rank_left_expdim": 30,  "rank_right_expdim": 30, "is_expand": False, "vocab_size": 30522, "hidden_size" : 768, "requires_grad": True, "num_hidden_layers": 12, "ops": "san_ffn"}

config = DottableDict(config)



def test_error(path,data,config):
    tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=True)
    raw_model = BertForPreTraining.from_pretrained(path) # BertForPreTraining.from_scratch(args.student_model)
    new_model = BertForPreTraining.from_pretrained(path)
    new_model.swap()
    # new_model.bert.swap(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_model = raw_model.to(device)
    new_model = new_model.to(device)
    new_model.eval()
    raw_model.eval()
    log_model(raw_model)
    log_model(new_model)

    # print(raw_model.bert.embeddings.word_embeddings.weight)
    # print(new_model.bert.embeddings.word_embeddings.weight)


    for epoch in range(1):
        epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=data, tokenizer=tokenizer,
                                                    num_data_epochs=1, reduce_memory=False)
        train_sampler = RandomSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=1)
        # with tqdm(total=len(train_dataloader), desc="Epoch {}".format(epoch)) as pbar:
        diff = []
        losses = []
        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # print(student_model.bert.embeddings.frequency_emb)
            input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch

            loss1 = raw_model(input_ids, segment_ids, input_mask, lm_label_ids, is_next).item()
            # exit()
            print("----")
            # new_model.step()
            loss2 = new_model(input_ids, segment_ids, input_mask, lm_label_ids, is_next).item()
            print(loss1, loss2)
            # diff.append(math.fabs(loss1-loss2)/math.sqrt(loss1 * loss2))
            # losses.append(loss2)
            # if step > 100:
            #     break
        # print(np.array(diff).mean())
        # print(np.array(losses).mean())

test_error(path,data,config)