# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import time
import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from .file_utils import WEIGHTS_NAME, CONFIG_NAME

from tensorly.decomposition import tucker
import tensorly as tl
import tqdm
from .embedding_utils import getTTEmbedding
from .embedding_utils import EmbeddingKet, EmbeddingKetXS
tl.set_backend('pytorch')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "",
}

BERT_CONFIG_NAME = 'bert_config.json'
TF_WEIGHTS_NAME = 'model.ckpt'


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
              "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                try:
                    pointer = getattr(pointer, 'bias')
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}
NORM = {'layer_norm': BertLayerNorm}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 pre_trained='',
                 training='',
                 attention_type='',
                 kernel_size=20):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                                                               and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.pre_trained = pre_trained
            self.training = training
            self.attention_type = attention_type
            self.kernel_size = kernel_size
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

def EmbeddingFactory(config):
    if "embedding_type" in config.__dict__ and config.embedding_type == "ket":
        word_embeddings = EmbeddingKet(config.vocab_size, config.hidden_size, padding_idx=0)
    elif "embedding_type" in config.__dict__ and config.embedding_type == "ketxs":
        word_embeddings = EmbeddingKetXS(config.vocab_size, config.hidden_size, padding_idx=0)
    elif "embedding_type" in config.__dict__ and config.embedding_type == "tt":
        word_embeddings = getTTEmbedding(config.vocab_size, config.hidden_size, padding_idx=0,embedding_type = "tt",d=3,rank = 8)
    elif "embedding_type" in config.__dict__ and config.embedding_type == "tr":
        word_embeddings = getTTEmbedding(config.vocab_size, config.hidden_size, padding_idx=0,embedding_type = "tr",d=3,rank = 8)
    else:
        word_embeddings =  nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
    return word_embeddings

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        # EmbeddingKet, EmbeddingKetXS
        if not config.load_compressed_model:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        else:
            self.word_embeddings = EmbeddingFactory(config)


        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def get_weights(self):
        if type(self.word_embeddings) ==  nn.Embedding:
            return self.word_embeddings.weight
        else:
            return self.word_embeddings.get_weights()

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        # print("The input ids shape is {}".format(input_ids.shape))
        # print("The word embedding shape is {}".format(words_embeddings.shape))
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Transformer
class BertSelfAttention(nn.Module):  # not intilized dense if use compresstion
    def __init__(self, config, layer_id = None):
        super(BertSelfAttention, self).__init__()
        self.layer_id = layer_id
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        if not config.load_compressed_model or (config.load_compressed_model and "san" not in config.ops):
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, output_att=False, weights = None):

        if weights is not None:
            mixed_query_layer,mixed_key_layer,mixed_value_layer = weights.qkv_transform(hidden_states, self.layer_id)
        else:
            mixed_query_layer = self.query(hidden_states) #
            mixed_key_layer = self.key(hidden_states) #
            mixed_value_layer = self.value(hidden_states)
        #print(temp,mixed_value_layer.sum())
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores


class BertAttention(nn.Module):
    def __init__(self, config, layer_id = None):
        super(BertAttention, self).__init__()
        self.layer_id = layer_id
        self.self = BertSelfAttention(config,layer_id= layer_id)
        self.output = BertSelfOutput(config,layer_id= layer_id)

    def forward(self, input_tensor, attention_mask,weights=None):
        self_output, layer_att = self.self(input_tensor, attention_mask, weights=weights)
        attention_output = self.output(self_output, input_tensor, weights=weights)

        return attention_output, layer_att


class BertSelfOutput(nn.Module):  # not intilized dense if use compresstion
    def __init__(self, config,layer_id = None):
        super(BertSelfOutput, self).__init__()
        self.layer_id = layer_id
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if not config.load_compressed_model or (config.load_compressed_model and "san" not in config.ops):
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, input_tensor, weights=None):

        if weights is not None :
            hidden_states = weights.output_transform(hidden_states, self.layer_id )
        else:
            hidden_states = self.dense(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertIntermediate(nn.Module): # not intilized dense if use compresstion
    def __init__(self, config, intermediate_size=-1, layer_id = None):
        super(BertIntermediate, self).__init__()
        self.layer_id = layer_id
        if intermediate_size < 0:
            intermediate_size = config.intermediate_size
        if not config.load_compressed_model or (config.load_compressed_model and "ffn" not in config.ops):
            self.dense = nn.Linear(config.hidden_size, intermediate_size)

        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states,weights=None): #since it was once transposed


        if weights is not None:
            hidden_states = weights.ffn_inner_transform(hidden_states,self.layer_id)
        else:
            hidden_states = self.dense(hidden_states) # use weights


        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class BertOutput(nn.Module):  # not intilized dense if use compresstion
    def __init__(self, config, intermediate_size=-1, layer_id = None):
        super(BertOutput, self).__init__()
        self.layer_id = layer_id
        if intermediate_size < 0:
            intermediate_size = config.intermediate_size
        if not config.load_compressed_model or (config.load_compressed_model and "ffn" not in config.ops):
            self.dense = nn.Linear(intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, weights=None):
    # no need transpose since we did twice transpose before tucker decompositon, e.g. dense.weight (transpose) and another explicit transpose

        if weights is not None:
            hidden_states = weights.ffn_output_transform(hidden_states,self.layer_id)
        else:
            hidden_states = self.dense(hidden_states) # using weights
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config,layer_id = None):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config,layer_id= layer_id)
        self.intermediate = BertIntermediate(config,layer_id=layer_id)
        self.output = BertOutput(config, layer_id= layer_id)

    def forward(self, hidden_states, attention_mask,ops,weights = None):
        if weights is not None and "san" in ops:
            attention_output, layer_att = self.attention(
                hidden_states, attention_mask, weights=weights)
        else:
            attention_output, layer_att = self.attention(
                hidden_states, attention_mask)

        if  weights is not None and "ffn" in ops:
            intermediate_output = self.intermediate(attention_output,weights = weights)
            layer_output = self.output(intermediate_output, attention_output,weights = weights)
        else:
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)



        return layer_output, layer_att


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config, layer_id=i)
                                    for i in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, ops,weights=None):
        all_encoder_layers = []
        all_encoder_atts = []
        for i, layer_module in enumerate(self.layer):
            all_encoder_layers.append(hidden_states)

            if weights is not None:
                hidden_states, layer_att = layer_module(
                    hidden_states, attention_mask,ops,weights = weights)
            else:
                hidden_states, layer_att = layer_module(hidden_states, attention_mask,ops)
            all_encoder_atts.append(layer_att)
        all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_atts


class BertPooler(nn.Module):
    def __init__(self, config, recurs=None):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. "-1" refers to last layer
        pooled_output = hidden_states[-1][:, 0]

        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        # Need to unty it when we separate the dimensions of hidden and emb
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)


        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.


        if type(bert_model_embedding_weights) == torch.Tensor: #  means ket
            self.use_linear = False
            self.weight = bert_model_embedding_weights.transpose(-2,-1)

        else:  # torch.nn.parameter.Parameter for original bert embedding
            self.use_linear = True
            self.decoder = nn.Linear(bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0), bias=False)
            self.decoder.weight = bert_model_embedding_weights

        self.bias = nn.Parameter(torch.zeros(
        bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):

        hidden_states = self.transform(hidden_states)
        if not self.use_linear:
            hidden_states = hidden_states@self.weight + self.bias
        else:
            hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

    def swap(self, bert_model_embedding_weights):
        self.use_linear = False
        self.weight = bert_model_embedding_weights.transpose(-2,-1)


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear )): #nn.Embedding
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_scratch(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        resolved_config_file = os.path.join(
            pretrained_model_name_or_path, CONFIG_NAME)
        config = BertConfig.from_json_file(resolved_config_file)

        logger.info("Model config {}".format(config))
        model = cls(config, *inputs, **kwargs)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        # Load config
        config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.

        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(
                pretrained_model_name_or_path, WEIGHTS_NAME)
            logger.info("Loading model {}".format(weights_path))
            state_dict = torch.load(weights_path, map_location='cpu')

        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(
                pretrained_model_name_or_path, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'

        logger.info('loading model...')
        load(model, prefix=start_prefix)
        logger.info('done!')
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        return model


# class Base(nn.Module):

def TuckerWeightsFactory(config):
    if config.is_expand:
        return TuckerWeights_Plus(config)
    else:
        return TuckerWeights(config)


class TuckerWeights(nn.Module):
    def __init__(self, config): #
        super(TuckerWeights, self).__init__()
        try:
            self.ops = config.ops
        except:
            print("cannot find the ops in the config file")
            self.ops = "san_ffn"
        if "ffn" in self.ops and "san" in self.ops:
            self.blocks_num = 12
        elif "ffn" in self.ops:
            self.blocks_num = 8
        else:
            self.blocks_num = 4

        if "ffn" in self.ops and "san" in self.ops:
            self.bias_blocks_num = 12
        elif "ffn" in self.ops:
            self.bias_blocks_num = 5
        else:
            self.bias_blocks_num = 4

        print("ops {}; blocks_num {}; bias_blocks_num {}".format(self.ops, self.blocks_num, self.bias_blocks_num))
        self.factor_left = nn.Parameter(torch.zeros(config.hidden_size, config.rank_condim), requires_grad=config.requires_grad)
        self.factor_right = nn.Parameter(torch.zeros(config.rank_dim,config.hidden_size), requires_grad=config.requires_grad)
        self.factor_layer = nn.Parameter(torch.zeros(config.num_hidden_layers*self.blocks_num, config.rank_layer), requires_grad=config.requires_grad)
        self.core = nn.Parameter(torch.zeros(config.rank_layer, config.rank_condim, config.rank_dim))

        self.bias = nn.Parameter(torch.zeros(config.num_hidden_layers, self.bias_blocks_num, config.hidden_size))
        self.ranks = [config.rank_layer, config.rank_condim, config.rank_dim]
        self.hidden_size = config.hidden_size
        self.config = config

    def forward(self,keep_gloabl_weights = True):  # change the order
        if not keep_gloabl_weights:
            return torch.einsum("ikl,xi,yk,lz->xyz", [self.core, self.factor_layer, self.factor_left, self.factor_right]).view(-1, 12, self.hidden_size, self.hidden_size), self.bias

        core = torch.einsum("ikl,xi->xkl", [self.core, self.factor_layer]) # 144 * d * d   # d is the rank of D
        #print("using a efficient way")


         # need ffn or not
        return core.view(-1,self.blocks_num,core.size()[-1],core.size()[-2]), self.bias, self.factor_left, self.factor_right

    def init(self,keep_gloabl_weights = True):
        self.weights,_,_,_ = self.forward(keep_gloabl_weights)


    def l2_loss(self, lamba=1):
        self.paras = [self.core[:,-1,-1], self.factor_left[:,-1], self.factor_left[-1,:] ]
        loss = [ torch.norm(para) for para in self.paras ] # l2 norm in default
        return loss * loss


    def step(self):
        if self.config.rank_dim >2:
            self.core = nn.Parameter(self.core.data[:,:-1,:-1])
            self.factor_left = nn.Parameter(self.factor_left.data[:,:-1])
            self.factor_right = nn.Parameter(self.factor_right.data[:-1,:])
            self.config.rank_condim = self.config.rank_condim -1
            self.config.rank_dim = self.config.rank_condim
            self.ranks = [self.config.rank_layer, self.config.rank_condim, self.config.rank_dim]
            print("reduce rank from {} to {}".format(self.config.rank_dim+1, self.config.rank_dim))
        else:
            print("rank is too low, no need for reduction")



    def qkv_transform(self,hidden_states, layer):
        weight,bias = self.weights[layer], self.bias[layer]
                #print(hidden_states.size(),left.size(),weight[0].size(),right.size())
        mixed_query_layer = hidden_states @ self.factor_left @ (weight[0] @ self.factor_right) + bias[0] #
        mixed_key_layer = hidden_states @ self.factor_left @ (weight[1] @ self.factor_right) + bias[1]#
        mixed_value_layer = hidden_states @ self.factor_left @ (weight[2] @ self.factor_right) + bias[2]
        return mixed_query_layer,mixed_key_layer,mixed_value_layer

    def output_transform(self,hidden_states,layer):
        weight,bias = self.weights[layer], self.bias[layer]
        hidden_states = hidden_states @  self.factor_left @ (weight[3] @ self.factor_right) + bias[3]
        return hidden_states

    def ffn_inner_transform(self,hidden_states,layer):
        weight,bias = self.weights[layer], self.bias[layer]
        bias = bias[-5:-1].view(-1)
        rank = weight.size()[-1]

        right_temp =  weight[-8:-4].reshape(-1 ,weight.size()[-1]).transpose(-1,-2).view( rank,4,rank) @ self.factor_left.transpose(-1,-2)
        right_temp = right_temp.view( rank, 4 *  hidden_states.size()[-1])
        hidden_states = (hidden_states @  self.factor_right.transpose(-1,-2) @ right_temp)+ bias
        return hidden_states


    def ffn_output_transform(self,hidden_states,layer):
        weight,bias = self.weights[layer], self.bias[layer]
        weight = weight[-4:].reshape(-1,weight.size()[-1]) # 4d * d
        bias = bias[-1]
        right_temp = (weight @ self.factor_right)  # 4d*d d*D = 4d*D
        hidden_states = hidden_states.view(hidden_states.size()[0], hidden_states.size()[1], 4,
                                                   -1) @ self.factor_left  # b*4*D D*d = b*4*d
        hidden_states = hidden_states.view(hidden_states.size()[0], hidden_states.size()[1], -1) @ right_temp + bias  # b*4*d * 4d*D


        return hidden_states

    def del_weights(self, config, encoder):
        for l in range(config.num_hidden_layers):
            del encoder.layer[l].attention.self.query, encoder.layer[l].attention.self.key, encoder.layer[
                l].attention.self.value, encoder.layer[l].attention.output.dense
            del encoder.layer[l].intermediate.dense, encoder.layer[l].output.dense
            torch.cuda.empty_cache()

    def get_weights_from_encoder(self,config,encoder):
        outputs = []
        outputs_bias = []
        for l in range(config.num_hidden_layers):

            output,bias_w = [],[]
            if "san" in self.ops:
                query = encoder.layer[l].attention.self.query.weight.transpose(-2,-1).unsqueeze(0)
                # print("Original shape is")
                # print(encoder.layer[l].attention.self.query.weight.shape)
                # print("After transpose")
                # print(query.shape)
                key = encoder.layer[l].attention.self.key.weight.transpose(-2,-1).unsqueeze(0)
                value = encoder.layer[l].attention.self.value.weight.transpose(-2,-1).unsqueeze(0)
                W_O = encoder.layer[l].attention.output.dense.weight.transpose(-2,-1).unsqueeze(0)

                bias_Q = encoder.layer[l].attention.self.query.bias.unsqueeze(0)
                # print(bias_Q.shape)
                bias_K = encoder.layer[l].attention.self.key.bias.unsqueeze(0)
                bias_V = encoder.layer[l].attention.self.value.bias.unsqueeze(0)
                bias_W_O = encoder.layer[l].attention.output.dense.bias.unsqueeze(0)

                output.extend([query, key, value, W_O])
                bias_w.extend([bias_Q, bias_K, bias_V, bias_W_O])

                del encoder.layer[l].attention.self.query,encoder.layer[l].attention.self.key, encoder.layer[l].attention.self.value,encoder.layer[l].attention.output.dense
            if "ffn" in self.ops:
                W_inner = encoder.layer[l].intermediate.dense.weight.view(-1, config.hidden_size, config.hidden_size)
                # print("inner weight shape is {}".format(W_inner.shape))
                # print(encoder.layer[l].intermediate.dense.weight.type)
                W_output = encoder.layer[l].output.dense.weight.transpose(0, 1).view(-1, config.hidden_size, config.hidden_size)

                bias_W_inner = encoder.layer[l].intermediate.dense.bias.view(-1, config.hidden_size)
                bias_W_output = encoder.layer[l].output.dense.bias.unsqueeze(0)
                output.extend([W_inner, W_output])
                bias_w.extend([bias_W_inner, bias_W_output])

                del encoder.layer[l].intermediate.dense, encoder.layer[l].output.dense

            torch.cuda.empty_cache()


            output = torch.cat(output, dim=0)
            #output = torch.cat([query, key, value, W_O, W_inner, W_output], dim=0)
            bias_w = torch.cat(bias_w, dim=0)

            outputs.append(output)
            outputs_bias.append(bias_w)
        weights = torch.stack(outputs, dim=0)
        biasw = torch.stack(outputs_bias, dim=0)

        sizes = [-1, self.hidden_size, self.hidden_size]
        weights = weights.view(sizes)
        return weights, biasw


    def raw_initialize(self,config,encoder):
        weights, biasw = self.get_weights_from_encoder(config,encoder)


        sizes = [-1, self.hidden_size, self.hidden_size]
        weights = weights.view(sizes)#.transpose(-2,-1)  # do not transpose here please!!! waby 4-6

        self.core.data.copy_(weights)
        self.bias.data.copy_(biasw.data)
        #print([factor.size() for factor in factors])
        self.factor_layer.data.copy_( torch.diag(torch.ones(self.config.num_hidden_layers*self.blocks_num)) ) # 144
        self.factor_left.data.copy_(torch.diag(torch.ones(768)))
        self.factor_right.data.copy_(torch.diag(torch.ones(768)))
        y = torch.einsum("ikl,xi,yk,zl->xyz", [self.core, self.factor_layer, self.factor_left, self.factor_right])
        print( "again average square error during compression： {}".format(((y - weights)**2).mean().cpu().detach().numpy()))




    def init_weights(self, config, encoder, keep_head=False, keep_layer=True):
        #self.raw_initialize(config,encoder)
        #return
        weights, biasw  = self.get_weights_from_encoder(config,encoder)
        #.transpose(-2,-1)  # do not transpose here please!!! waby 4-6
        start = time.time()
        #print(weights.size())
        #print(self.ranks)
        if tl.__version__ == "0.5.1":
            core, factors = tucker(weights, rank=self.ranks,init='random',verbose =True)#,tol=1e-8,tol = 1e-8
        else:
            core, factors = tucker(weights, ranks=self.ranks,init='random',verbose =True)#,tol = 1e-8 ,tol = 1e-8
        print("compression from {} to {}: {} s".format(weights.size(), self.ranks, time.time()-start))
        y = torch.einsum("ikl,xi,yk,zl->xyz", [core, factors[0], factors[1], factors[2]])

        print( "again average l1 error during compression： {}".format((y - weights).abs().mean().cpu().detach().numpy()))

        self.core.data.copy_(core)
        self.bias.data.copy_(biasw.data)
        #print([factor.size() for factor in factors])
        self.factor_layer.data.copy_(factors[0])
        self.factor_left.data.copy_(factors[1])
        self.factor_right.data.copy_(factors[2].transpose(-1,-2))

        y = torch.einsum("ikl,xi,yk,lz->xyz", [self.core, self.factor_layer, self.factor_left, self.factor_right])

        print( "again average l1 error during compression： {}".format((y - weights).abs().mean().cpu().detach().numpy()))
        return weights


class TuckerWeights_Plus(TuckerWeights):
    def __init__(self, config):  #
        super(TuckerWeights_Plus, self).__init__(config)

        self.exp_left = nn.Parameter(torch.rand(config.num_hidden_layers * self.blocks_num, config.hidden_size, config.rank_left_expdim)/100,
                                     requires_grad=config.requires_grad)
        self.exp_right = nn.Parameter(torch.rand(config.num_hidden_layers * self.blocks_num, config.rank_right_expdim, config.hidden_size)/100,
                                     requires_grad=config.requires_grad)
        self.exp_core11= self.core
        self.exp_core12 = nn.Parameter(torch.rand(config.num_hidden_layers * self.blocks_num, config.rank_condim, config.rank_right_expdim)/100,
                                     requires_grad=config.requires_grad)
        self.exp_core21 = nn.Parameter(torch.rand(config.num_hidden_layers * self.blocks_num, config.rank_left_expdim, config.rank_dim)/100,
                                     requires_grad=config.requires_grad)
        self.exp_core22 = nn.Parameter(torch.rand(config.num_hidden_layers * self.blocks_num, config.rank_left_expdim, config.rank_right_expdim)/100,
                                     requires_grad=config.requires_grad)

    def forward(self,keep_gloabl_weights = True):  # change the order
        if not keep_gloabl_weights:
            return torch.einsum("ikl,xi,yk,lz->xyz", [self.core, self.factor_layer, self.factor_left, self.factor_right]).view(-1, 12, self.hidden_size, self.hidden_size), self.bias

        # core = torch.einsum("ikl,xi->xkl", [self.core, self.factor_layer]) # 144 * d * d   # d is the rank of D
        #print("using a efficient way")
        # w11 = core.view(-1,self.blocks_num,core.size()[-2],core.size()[-1])
        w11 = self.exp_core11.view(-1, self.blocks_num, self.exp_core11.size()[-2], self.exp_core11.size()[-1])
        w21 = self.exp_core21.view(-1, self.blocks_num, self.exp_core21.size()[-2], self.exp_core21.size()[-1])
        w12 = self.exp_core12.view(-1, self.blocks_num, self.exp_core12.size()[-2], self.exp_core12.size()[-1])
        w22 = self.exp_core22.view(-1, self.blocks_num, self.exp_core22.size()[-2], self.exp_core22.size()[-1])

        w_top = torch.cat((w11, w12), 3)
        w_dwn = torch.cat((w21, w22), 3)

        weights = torch.cat((w_top, w_dwn), 2)

        u2  = self.exp_left.view(-1, self.blocks_num, self.exp_left.size()[-2], self.exp_left.size()[-1])
        v2  = self.exp_right.view(-1, self.blocks_num, self.exp_right.size()[-2], self.exp_right.size()[-1])

         # need ffn or not
        return weights, u2, v2, \
               self.bias, self.factor_left, self.factor_right

    def init(self,keep_gloabl_weights = True):
        self.weights,self.u2, self.v2,_,_,_ = self.forward(keep_gloabl_weights)

    def qkv_transform(self,hidden_states, layer):
        weight,bias = self.weights[layer], self.bias[layer]
        u2, v2 = self.u2[layer], self.v2[layer]

        # mixed_query_layer = hidden_states @ torch.cat((self.factor_left, u2[0]), 1) @ (weight[0] @ torch.cat((self.factor_right, v2[0]), 0)) + bias[0]  #
        # mixed_key_layer   = hidden_states @ torch.cat((self.factor_left, u2[1]), 1) @ (weight[1] @ torch.cat((self.factor_right, v2[1]), 0)) + bias[1]  #
        # mixed_value_layer = hidden_states @ torch.cat((self.factor_left, u2[2]), 1) @ (weight[2] @ torch.cat((self.factor_right, v2[2]), 0)) + bias[2]  #

        mixed_query_layer = hidden_states @ (torch.cat((self.factor_left, u2[0]), 1) @ weight[0]).tanh() @ torch.cat((self.factor_right, v2[0]), 0) + bias[0]  #
        mixed_key_layer   = hidden_states @ (torch.cat((self.factor_left, u2[1]), 1) @ weight[1]).tanh() @ torch.cat((self.factor_right, v2[1]), 0) + bias[1]  #
        mixed_value_layer = hidden_states @ (torch.cat((self.factor_left, u2[2]), 1) @ weight[2]).tanh() @ torch.cat((self.factor_right, v2[2]), 0) + bias[2]  #

        return mixed_query_layer,mixed_key_layer,mixed_value_layer

    def output_transform(self,hidden_states,layer):
        weight,bias = self.weights[layer], self.bias[layer]
        u2, v2 = self.u2[layer], self.v2[layer]

        hidden_states = hidden_states @ (torch.cat((self.factor_left, u2[3]), 1) @ weight[3]).tanh() @ torch.cat((self.factor_right, v2[3]), 0) + bias[3]  #
        return hidden_states

    def ffn_inner_transform(self,hidden_states,layer):
        weight,bias = self.weights[layer], self.bias[layer]
        u2, v2 = self.u2[layer], self.v2[layer]
        bias = bias[4:8].view(-1)

        w4 = (torch.cat((self.factor_left, u2[4]), 1) @ weight[4]).tanh() @ torch.cat((self.factor_right, v2[4]), 0)
        w5 = (torch.cat((self.factor_left, u2[5]), 1) @ weight[5]).tanh() @ torch.cat((self.factor_right, v2[5]), 0)
        w6 = (torch.cat((self.factor_left, u2[6]), 1) @ weight[6]).tanh() @ torch.cat((self.factor_right, v2[6]), 0)
        w7 = (torch.cat((self.factor_left, u2[7]), 1) @ weight[7]).tanh() @ torch.cat((self.factor_right, v2[7]), 0)

        wa = torch.cat((w4, w5, w6, w7), 0)
        hidden_states = hidden_states @ wa.transpose(-1,-2) + bias

        return hidden_states


    def ffn_output_transform(self,hidden_states,layer):
        weight,bias = self.weights[layer], self.bias[layer]
        u2, v2 = self.u2[layer], self.v2[layer]
        # weight = weight[8:].reshape(-1,weight.size()[-1]) # 4d * d
        bias = bias[8]

        w8  = (torch.cat((self.factor_left, u2[8]), 1)  @ weight[8]).tanh()  @ torch.cat((self.factor_right, v2[8]), 0)
        w9  = (torch.cat((self.factor_left, u2[9]), 1)  @ weight[9]).tanh()  @ torch.cat((self.factor_right, v2[9]), 0)
        w10 = (torch.cat((self.factor_left, u2[10]), 1) @ weight[10]).tanh() @ torch.cat((self.factor_right, v2[10]), 0)
        w11 = (torch.cat((self.factor_left, u2[11]), 1) @ weight[11]).tanh() @ torch.cat((self.factor_right, v2[11]), 0)

        wa2 = torch.cat((w8, w9, w10, w11), 0)
        hidden_states = hidden_states @ wa2 + bias

        return hidden_states

    def init_weights(self, config, encoder):
        # self.raw_initialize(config,encoder)
        # return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = super(TuckerWeights_Plus, self).init_weights(config, encoder).detach().clone()
        weights = weights.to(device)
        # weights = torch.tensor(weights, requires_grad=False)#.detach()

        # optimization hyper-parameters setting
        n_iter = 20000
        lr = 0.005
        penalty = 0.000005

        # w11 = torch.randn(self.core.size(), device=device, requires_grad=True)
        core = self.core.data.detach().clone()
        core = core.to(device)

        factor_layer = self.factor_layer.data.detach().clone()
        factor_layer = factor_layer.to(device)

        w11 = torch.einsum("ikl,xi->xkl", [core, factor_layer])

        # w11 = self.core.data.detach().clone()
        w11 = w11.to(device)

        u1 = self.factor_left.data.detach().clone()
        u1 = u1.to(device)

        v1 = self.factor_right.data.detach().clone()
        v1 = v1.to(device)

        u2 = self.exp_left.data.detach().clone()
        u2 = u2.to(device)

        v2 = self.exp_right.data.detach().clone()
        v2 = v2.to(device)

        w12 = self.exp_core12.data.detach().clone()
        w12 = w12.to(device)

        w21 = self.exp_core21.data.detach().clone()
        w21 = w21.to(device)

        w22 = self.exp_core22.data.detach().clone()
        w22 = w22.to(device)

        weights = weights.detach().clone()
        weights = weights.to(device)

        ## test the reconstruction error
        rec11 = torch.matmul(torch.matmul(u1, w11), v1)
        rec_error = torch.norm(rec11 - weights) / torch.norm(weights)
        print("Rec. error: {}".format(rec_error))

        # factors_stage1 = [u1, v1, core, factor_layer]
        factors_stage2 = [u2, v2, w12, w21, w22, w11, u1, v1] #+ factors_stage1

        for f in factors_stage2:
            f.requires_grad = True

        pp = 0
        for p in factors_stage2:
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn

        num_par = 0
        for f in factors_stage2:
            num = 1
            for i in list(f.size()):
                num = num * i
            num_par = num_par + num

        num_par_ori = 1
        for i in list(weights.size()):
            num_par_ori = num_par_ori * i

        print("Compression ratio: {}".format(num_par_ori / num_par))

        # optimizer_stage1 = torch.optim.Adam(factors_stage1, lr=lr)
        optimizer_stage2 = torch.optim.Adam(factors_stage2, lr=lr)

        for i in range(1, n_iter):

            optimizer_stage2.zero_grad()
            rec11 = torch.matmul(torch.matmul(u1, w11).tanh(), v1)
            rec12 = torch.matmul(torch.matmul(u1, w12).tanh(), v2)
            rec21 = torch.matmul(torch.matmul(u2, w21).tanh(), v1)
            rec22 = torch.matmul(torch.matmul(u2, w22).tanh(), v2)

            # w11 = torch.einsum("ikl,xi->xkl", [core, factor_layer])
            # rec11 = torch.matmul(u1, torch.matmul(w11, v1))
            # rec12 = torch.matmul(u1, torch.matmul(w12, v2))
            # rec21 = torch.matmul(u2, torch.matmul(w21, v1))
            # rec22 = torch.matmul(u2, torch.matmul(w22, v2))

            rec = rec11 + rec12 + rec21 + rec22

            # squared l2 loss
            loss = torch.norm(rec - weights)
            # loss = torch.sum(torch.abs(rec - tensor))
            loss1 = loss
            # squared l2 penalty on the factors of the decomposition
            for f in factors_stage2:
                loss = loss + penalty * f.pow(2).sum()

            loss.backward()
            optimizer_stage2.step()

            if i % 500 == 0:
                rec_error_norm = torch.norm(rec - weights) / torch.norm(weights)
                rec_error_mean = torch.mean(torch.abs(rec - weights))
                rec_error_sum = torch.sum(torch.abs(rec - weights))
                rec_error_max = torch.max(torch.abs(rec - weights))
                print("Epoch {},. Rec. error: {} . Rec. error: {} . Rec. error: {} . Rec. error: {}".format(i,
                                                rec_error_norm, rec_error_mean, rec_error_sum, rec_error_max))
                print("Epoch {},. loss1: {}, loss2: {}".format(i, loss1, loss - loss1))

        with torch.no_grad():
            # self.core.data.copy_(core)
            # self.factor_layer.copy_(factor_layer)
            self.factor_left.copy_(u1)
            self.factor_right.copy_(v1)

            self.exp_left.data.copy_(u2)
            self.exp_right.data.copy_(v2)
            self.exp_core11.data.copy_(w11)
            self.exp_core12.data.copy_(w12)
            self.exp_core21.data.copy_(w21)
            self.exp_core22.data.copy_(w22)

        # w11 = torch.einsum("ikl,xi->xkl", [core, factor_layer])
        # rec11 = torch.matmul(u1, torch.matmul(w11, v1))
        # rec12 = torch.matmul(u1, torch.matmul(w12, v2))
        # rec21 = torch.matmul(u2, torch.matmul(w21, v1))
        # rec22 = torch.matmul(u2, torch.matmul(w22, v2))
        #
        # print("again average l1 error during compression： {}".format(
        #     (rec11 + rec12 + rec21 + rec22 - weights).abs().mean().cpu().detach().numpy()))
        #
        # y11 = torch.einsum("ikl,xi,yk,lz->xyz", [self.core, self.factor_layer, self.factor_left, self.factor_right])
        y11 = torch.matmul(torch.matmul(self.factor_left, self.exp_core11).tanh(), self.factor_right)
        y12 = torch.matmul(torch.matmul(self.factor_left, self.exp_core12).tanh(), self.exp_right)
        y21 = torch.matmul(torch.matmul(self.exp_left, self.exp_core21).tanh(), self.factor_right)
        y22 = torch.matmul(torch.matmul(self.exp_left, self.exp_core22).tanh(), self.exp_right)
        weights = weights.to('cpu')

        print("again average l1 error during compression： {}".format((y11 + y12 + y21 + y22 - weights).abs().mean().cpu().detach().numpy()))


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.ops = config.ops

        self.pooler = BertPooler(config)
        self.encoder = BertEncoder(config)

        self.apply(self.init_bert_weights)
        self.attention_type = config.attention_type
        self.config = config
        #self.swap(config) # may call it latter, since it may have first load paramters for a noraml BERT and then compress it
        self.use_compression = False
        if "load_compressed_model" in config.__dict__ and config.load_compressed_model:
            self.tucker_weighter = TuckerWeightsFactory(config)
            self.use_compression = True
            #bert_model_embedding_weights = self.embeddings.word_embeddings.get_weights()
            #self.cls.predictions.swap(word_embeddings_weights)


    def swap(self,config =None,process_embedding=False):
        if config is None:
            config = self.config

        if process_embedding: # this implementations is wrong
            try:
                bert_model_embedding_weights = self.embeddings.word_embeddings.get_weights() # cold embeddings
            except:
                bert_model_embedding_weights  =   self.embeddings.word_embeddings.weight

            self.embeddings.word_embeddings = EmbeddingFactory(config)


            if type(self.embeddings.word_embeddings) != nn.Embedding:
                if "compression_from_scratch" in config.__dict__ and  config.compression_from_scratch:
                    pass
                else:
                    print("init embedding weights  from a pretrained model ")
                    self.embeddings.word_embeddings.initialize(bert_model_embedding_weights,steps=10)

        if "rank_dim" in config.__dict__ and "rank_layer" in config.__dict__ and "rank_condim" in config.__dict__:
            print("using conpression during model building")
            config.use_compression = True
            self.tucker_weighter = TuckerWeightsFactory(config)
            if "compression_from_scratch" in config.__dict__ and   config.compression_from_scratch:
                self.tucker_weighter.del_weights(config, self.encoder)
            else:
                print("init weights (including core, factors, and bias) from a pretrained ")

                self.tucker_weighter.init_weights(config, self.encoder)

        else:
            config.use_compression = False

        self.use_compression = config.use_compression

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=True, output_att=False):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # transformer mask
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        try:
            dtype = next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            dtype = first_tuple[1].dtype

        extended_attention_mask = extended_attention_mask.to(dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        if self.use_compression:
            weights = self.tucker_weighter(keep_gloabl_weights = True) # [weights, bias,left,right]
            #weights = self.tucker_weighter(keep_gloabl_weights = False)
            self.tucker_weighter.init()
        else:
            self.tucker_weighter = None

        encoded_layers, layer_atts = self.encoder(embedding_output,
                                                  extended_attention_mask, self.ops,weights=self.tucker_weighter)

        pooled_output = self.pooler(encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        if not output_att:
            return encoded_layers, pooled_output

        return encoded_layers, layer_atts, pooled_output


class BertForPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)

        word_embeddings_weights = self.bert.embeddings.get_weights()


        self.cls = BertPreTrainingHeads(config, word_embeddings_weights)
        self.apply(self.init_bert_weights)
        self.config = config

        if "load_compressed_model" in config.__dict__ and config.load_compressed_model == True:
            if "embedding_type"  in config.__dict__ and config.embedding_type != "other":
                word_embeddings_weights = self.bert.embeddings.get_weights()
                self.cls.predictions.swap(word_embeddings_weights)
    def swap(self,config = None):
        if config is None:
            config = self.config

        self.bert.swap(config)

        if "embedding_type"  in config.__dict__ and config.embedding_type != "other":
            word_embeddings_weights = self.bert.embeddings.get_weights()
            self.cls.predictions.swap(word_embeddings_weights)

            #self.apply(self.init_bert_weights)

    def step(self):
        try:
            self.bert.tucker_weighter.step()
        except Exception as e:
            print(e)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, next_sentence_label=None):
        #x = self.bert(input_ids, token_type_ids, attention_mask,output_all_encoded_layers=False)

        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        elif masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss = masked_lm_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertForMaskedLM(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                output_att=False, infer=False):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=True, output_att=output_att)

        if output_att:
            sequence_output, att_output = sequence_output
        prediction_scores = self.cls(sequence_output[-1])

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            if not output_att:
                return masked_lm_loss
            else:
                return masked_lm_loss, att_output
        else:
            if not output_att:
                return prediction_scores
            else:
                return prediction_scores, att_output


class BertForNextSentencePrediction(BertPreTrainedModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls(pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForSentencePairClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSentencePairClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, a_input_ids, b_input_ids, a_token_type_ids=None, b_token_type_ids=None,
                a_attention_mask=None, b_attention_mask=None, labels=None):
        _, a_pooled_output = self.bert(
            a_input_ids, a_token_type_ids, a_attention_mask, output_all_encoded_layers=False)
        # a_pooled_output = self.dropout(a_pooled_output)

        _, b_pooled_output = self.bert(
            b_input_ids, b_token_type_ids, b_attention_mask, output_all_encoded_layers=False)
        # b_pooled_output = self.dropout(b_pooled_output)

        logits = self.classifier(torch.relu(torch.cat((a_pooled_output, b_pooled_output,
                                                       torch.abs(a_pooled_output - b_pooled_output)), -1)))

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # self.init_weights()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, head_mask=None, labels=None):

        # outputs = self.bert(input_ids, attention_mask=attention_mask,  token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        outputs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True, output_att=True)
        # encoded_layers, layer_atts, pooled_output
        pooled_output = outputs[-1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[:2]  # add hidden states and attention if they are here
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                # print(logits)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForQuestionAnswering(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        loss, start_scores, end_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)

        self.num_labels = 2 #config.num_labels

        self.bert = BertModel(config)

        self.qa_outputs = nn.Linear(config.hidden_size,self.num_labels)

        # self.init_weights()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):

        outputs = self.bert(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            # head_mask=head_mask
                            output_all_encoded_layers = False,
                            )
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = start_logits, end_logits# + outputs[2:]

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            # outputs = (total_loss,) + outputs
            outputs = total_loss
        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

    def save_pretrained(self, save_directory_or_file, epoch=0):
        """ Save a model card object to the directory or file `save_directory_or_file`.
        """
        if os.path.isdir(save_directory_or_file):
            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_card_file = os.path.join(save_directory_or_file, MODEL_CARD_NAME)
        else:
            output_model_card_file = save_directory_or_file
        if epoch != 0:
            save_directory_or_file + "_" + str(epoch)
        self.to_json_file(output_model_card_file)
        logger.info("Model card saved in {}".format(output_model_card_file))

    def save_model(self, save_directory_or_file, tokenizer, epoch=0):
        model_name = WEIGHTS_NAME
        output_model_file = os.path.join(save_directory_or_file, model_name)
        output_config_file = os.path.join(save_directory_or_file, CONFIG_NAME)

        torch.save(self.state_dict(), output_model_file)
        self.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(save_directory_or_file)
        logger.info("Model saved in {}".format(output_model_file))
        logger.info("config card saved in {}".format(output_config_file))
        logger.info("tokenizer saved in {}".format(save_directory_or_file))