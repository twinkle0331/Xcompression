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
# from transformer.modeling import BertForPreTraining,BertForSequenceClassification
from transformer.tokenization import BertTokenizer
# import torchvision
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
        print('n: {}'.format(n))
        print('p: {}'.format(p.nelement()))
        size += p.nelement()
    print('Total parameters: {}'.format(size))

path = "models/bert-base-uncased" # change load_compressed_model as True please.
data = Path("data/pregenerated_data")


def test_error(path,data):
    tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=True)
    model = BertForPreTraining.from_pretrained(path) # BertForPreTraining.from_scratch(args.student_model)
    # model.swap()
    # exit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    log_model(model)


    # print(raw_model.bert.embeddings.word_embeddings.weight)
    # print(new_model.bert.embeddings.word_embeddings.weight)


    for epoch in range(1):
        epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=data, tokenizer=tokenizer,
                                                    num_data_epochs=1, reduce_memory=False)
        train_sampler = RandomSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=2)
        # with tqdm(total=len(train_dataloader), desc="Epoch {}".format(epoch)) as pbar:
        diff = []
        losses = []
        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
            loss = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next)
            print(loss)
test_error(path,data)