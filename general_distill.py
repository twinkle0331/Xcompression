
from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import time
import json
import torch

use_hvd = True
try:
    import horovod.torch  as hvd
    if use_hvd == True:
        hvd.init()
        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())
        print("init with {} local ranks".format(hvd.local_rank()))
except Exception as e:
    print(e)
    use_hvd = False


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
from tqdm import tqdm,trange
import torch.multiprocessing as mp
from torch.nn import MSELoss
from tempfile import TemporaryDirectory

from torch.utils.data import (DataLoader,RandomSampler,Dataset)
from torch.utils.data.distributed import DistributedSampler

from transformer.optimization import BertAdam
from transformers import BertTokenizer
from transformer.modeling import BertForSequenceClassification,BertModel,BertForPreTraining
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pathlib import Path
from collections import namedtuple

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")
def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    if len(tokens) > max_seq_length:
        print('len(tokens): {}'.format(len(tokens)))
        print('tokens: {}'.format(tokens))
        tokens = tokens[:max_seq_length]

    if len(tokens) != len(segment_ids):
        print('tokens: {}\nsegment_ids: {}'.format(tokens, segment_ids))
        segment_ids = [0] * len(tokens)

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = int(epoch % num_data_epochs)
        print('training_path: {}'.format(training_path))
        data_file = training_path / "epoch_{}.json".format(self.data_epoch)
        metrics_file = training_path / "epoch_{}_metrics.json".format(self.data_epoch)

        print('data_file: {}'.format(data_file))
        print('metrics_file: {}'.format(metrics_file))

        if not data_file.is_file():
            data_file = training_path / "epoch_{}.json".format(self.data_epoch + 10)
            metrics_file = training_path / "epoch_{}_metrics.json".format(self.data_epoch + 10)
            print('data_file: {}'.format(data_file))
            print('metrics_file: {}'.format(metrics_file))
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None

        if use_hvd:
            num_samples = int(num_samples / hvd.size())
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path('/cache/')
            input_ids = np.memmap(filename=self.working_dir / 'input_ids_{}.memmap'.format(hvd.local_rank()),
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir / 'input_masks_{}.memmap'.format(hvd.local_rank()),
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir / 'segment_ids_{}.memmap'.format(hvd.local_rank()),
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir / 'lm_label_ids_{}.memmap'.format(hvd.local_rank()),
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir / 'is_nexts_{}.memmap'.format(hvd.local_rank()),
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)

        print("Loading training examples for epoch {}".format(epoch))

        with data_file.open() as f:
            j = 0
            for i, line in enumerate(
                    tqdm(f, total=num_samples, desc="Training examples", mininterval=120, maxinterval=300)):
                if i % hvd.size() != hvd.rank():
                    continue
                if j >= len(input_ids):
                    break
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[j] = features.input_ids
                segment_ids[j] = features.segment_ids
                input_masks[j] = features.input_mask
                lm_label_ids[j] = features.lm_label_ids
                is_nexts[j] = features.is_next
                j += 1
            # Split the data to different processes

        # assert i == num_samples - 1  # Assert that the sample count metric was true
        print("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(int(self.is_nexts[item])))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pregenerated_data",
                        type=Path,
                        required=True)
    parser.add_argument("--teacher_model",
                        default="models/bert-base-uncased",
                        type=str,
                        required=True)
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        required = False,
                        help = "The maximum total input sequence after WordPiece tokernization. \n"
                               "Sequences longer than this will be truncated,and sequences shorter \n"
                               "than this will be padded")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True
                        )
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        required= False,
                        help="The initial learning rate for Adam")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        required = False,
                        help ="Total number of training epoch to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        required = False,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        required = False,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--eval_step',
                        type=int,
                        required = False,
                        default=1000)
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-2,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--reduce_memory",
                        action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--use_swap',
                        action='store_true',
                        help="Whether to swap the student model")
    parser.add_argument('--mp', type=bool, default=False, help='Multiprocess distributed mode')

    # add for huawei cloud
    parser.add_argument("--data_url", type=str, default=None, help="s3 url")
    parser.add_argument("--train_url", type=str, default=None, help="s3 url")
    parser.add_argument("--init_method", default='', type=str)

    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    print("args:{}".format(args))

    samples_per_epoch = []
    num_train_epochs = args.num_train_epochs
    for i in range(int(num_train_epochs)):
        epoch_file = args.pregenerated_data / "epoch_{}.json".format(i)
        metrics_file = args.pregenerated_data / "epoch_{}_metrics.json".format(i)
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics["num_training_examples"])
        elif (args.pregenerated_data / "epoch_{}.json".format(i+10)).is_file():
            epoch_file = args.pregenerated_data / "epoch_{}.json".format(i+10)
            metrics_file =args.pregenerated_data / "epoch_{}_metrics.json".format(i+10)
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics["num_training_examples"])
        else:
            if i == 0:
                exit("No training data was found!")
            print("Warning! There are fewer epochs of pregenerated data ({}) than training epochs ({}).".format(i,
                                                                                                                args.num_train_epochs))
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = args.num_train_epochs

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.output_dir) and hvd.rank() == 0:
        if (use_hvd and hvd.rank() == 0) or (not use_hvd and args.local_rank in [-1, 0]):
            os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)

    total_train_examples = 0
    for i in range(int(args.num_train_epochs)):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    if use_hvd:
        num_train_optimization_steps = int(
            total_train_examples / args.train_batch_size / args.gradient_accumulation_steps / hvd.size())
    else:
        num_train_optimization_steps = int(
            total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)

    teacher_model = BertModel.from_pretrained(args.teacher_model)
    student_model = BertModel.from_pretrained(args.student_model)
    if args.use_swap:
        student_model.swap()

    student_model.to(device)
    teacher_model.to(device)

    size = 0
    for n,p in student_model.named_parameters():
        print("n:{}".format(n))
        #print("p:{}".format(p))
        size += p.nelement()

    print("Total parameters:{}".format(size))

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(student_model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    schedule = 'warmup_linear'


    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)
    loss_mse = MSELoss()

    if use_hvd:
        hvd.broadcast_parameters(student_model.state_dict(), root_rank=0)
        optimizer = hvd.DistributedOptimizer(optimizer, student_model.named_parameters(),
                                             backward_passes_per_step=args.gradient_accumulation_steps)
        print("broadcast over")

    global_step = 0
    print("***** Running training *****")
    print("  Num examples = {}".format(total_train_examples))
    print("  Batch size = {}".format(args.train_batch_size))
    print("  Num steps = {}".format(num_train_optimization_steps))

    for epoch in trange(int(args.num_train_epochs),desc="Epoch"):
        epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                            num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)

        if use_hvd:
            train_sampler = torch.utils.data.distributed.DistributedSampler(epoch_dataset, num_replicas=1, rank=0)
        else:
            train_sampler = RandomSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset,sampler=train_sampler,batch_size=args.train_batch_size)

        tr_loss = 0.
        tr_att_loss = 0.
        tr_rep_loss = 0.
        student_model.train()
        nb_tr_examples, nb_tr_steps = 0,0
        with tqdm(total=len(train_dataloader),desc="Epoch {}".format(epoch)) as pbar:
            for step,batch in enumerate(tqdm(train_dataloader, desc="Iteration",mininterval=120, maxinterval=300, ascii=True)):
            # for step,batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids,input_mask,segment_ids,lm_label_ids,is_next = batch

                if input_ids.size()[0] != args.train_batch_size:
                    continue

                att_loss = 0.
                rep_loss = 0.

                student_reps, student_atts,_ = student_model(input_ids, segment_ids, input_mask,output_att=True)
                with torch.no_grad():
                    teacher_reps, teacher_atts,_ = teacher_model(input_ids, segment_ids, input_mask,output_att=True)
                teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]  # speedup 1.5x
                teacher_atts = [teacher_att.detach() for teacher_att in teacher_atts]

                new_student_rep = student_reps[-1]
                new_teacher_rep = teacher_reps[-1]
                teacher_layer_num = len(new_teacher_rep)
                student_layer_num = len(new_student_rep)
                assert teacher_layer_num % student_layer_num == 0

                new_student_att = student_atts[-1]
                new_teacher_att = teacher_atts[-1]

                for student_att, teacher_att in zip(new_student_att, new_teacher_att):
                    student_att = torch.where(student_att <= -1e2,
                                              torch.zeros_like(student_att).cuda(non_blocking=True),
                                              student_att)
                    teacher_att = torch.where(teacher_att <= -1e2,
                                              torch.zeros_like(teacher_att).cuda(non_blocking=True),
                                              teacher_att)
                    att_loss += loss_mse(student_att, teacher_att)

                for student_rep, teacher_rep in zip(new_student_rep, new_teacher_rep):
                    rep_loss += loss_mse(student_rep, teacher_rep)

                loss = att_loss + rep_loss

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_att_loss += att_loss.item()
                tr_rep_loss += rep_loss.item()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                mean_att_loss = tr_att_loss * args.gradient_accumulation_steps / nb_tr_steps
                mean_rep_loss = tr_rep_loss * args.gradient_accumulation_steps / nb_tr_steps

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if (global_step + 1) % args.eval_step == 0 and hvd.rank()==0:
                        nb_tr_steps = 0
                        tr_loss = 0
                        tr_rep_loss = 0
                        tr_att_loss = 0
                        result = {}
                        result['global_step'] = global_step
                        result['loss'] = mean_loss
                        result['att_loss'] = mean_att_loss
                        result['rep_loss'] = mean_rep_loss
                        output_eval_file = os.path.join(args.output_dir, "log.txt")
                        with open(output_eval_file, "a") as writer:
                            print("***** Eval results *****")
                            for key in sorted(result.keys()):
                                print("  %s = %s", key, str(result[key]))
                                writer.write("%s = %s\n" % (key, str(result[key])))

                        # Save a trained model
                        model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                        print("** ** * Saving fine-tuned model ** ** * ")
                        # Only save the model it-self
                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

                        output_model_file = os.path.join(args.output_dir, model_name)
                        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir)

            if hvd.rank() == 0:
                model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                logging.info("** ** * Saving fine-tuned model ** ** * ")
                model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                # save other states
                other_state = {
                    'optimizer': optimizer.state_dict(),
                    'global_step': global_step
                }
                torch.save(other_state, args.output_dir + '/ckpt.t7')

                output_model_file = os.path.join(args.output_dir, model_name)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(args.output_dir)


if __name__ == "__main__":
    main()

