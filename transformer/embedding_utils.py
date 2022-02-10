import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging


import time
from functools import reduce
try:
    from gpytorch.lazy import KroneckerProductLazyTensor, NonLazyTensor
except:
    # raise Exception("You should install GPyTorch. \n"
    #                 + "$ conda install gpytorch -c gpytorch \n"
    #                 + "https://github.com/cornellius-gp/gpytorch \n"
    #                 )
    print("cannot install gpytorch because potroch version is lower than 1.7")


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


class BaseEmbedding(nn.Embedding):
    def __init__(self):
        super(nn.Embedding, self).__init__()

    def get_weights(self):
        return None

    def initialize(self, embeddings, steps=1000):
        if type(embeddings) != torch.Tensor:
            embeddings = torch.Tensor(embeddings) #.cuda
        # print([p.size() for p in self.parameters()])
        optimizer = torch.optim.SGD(self.parameters(), lr=1.0)
        es = EarlyStopping(patience=6)
        for i in range(steps):
            # print(self.get_weights())
            loss = torch.mean((self.get_weights() - embeddings)**2)
            #loss = torch.mean(torch.abs(self.get_weights() - embeddings) )
            loss.backward(retain_graph=True)
            if es.step(loss):
                print("early stoped")
                break
            print("compressing word embeddings with loss {}".format(loss.item()))
            optimizer.step()


class EmbeddingKet(BaseEmbedding):
    r"""This is a new embedding using Kronecker products.
    Order = order + 1
    """
    __constants__ = ['num_embeddings', 'embedding_dim', 'order', 'rank', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    def __init__(self, num_embeddings, embedding_dim, order=2, rank=1, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        """
            order: number of times we do the tensor product
            rank : Rank of the matrix, the dimension that we calcualte the sum of batches
        """
        super(EmbeddingKet, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_dim_leaf = math.ceil((embedding_dim) ** (1 / order))
        logging.info('EmbeddingKet base num_embeddings: ' + str(self.num_embeddings))
        logging.info('EmbeddingKet embedding_dim_leaf: ' + str(self.embedding_dim_leaf))
        self.rank = rank
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.order = order
        if _weight is None:
            # Ali: Creating Leaf Weights for Tensor product
            self.weight_leafs = nn.Parameter(torch.Tensor(
                # TODO:  During backward do we need all of these parameters to be loaded ? can we do better?
                self.order, self.rank, self.num_embeddings, self.embedding_dim_leaf))
            logging.info('EmbeddingKet weight_leafs shape: ' + str(self.weight_leafs.shape))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [self.order, self.rank, self.num_embeddings_leaf, self.embedding_dim_leaf], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight_leafs = nn.Parameter(_weight)
        self.sparse = sparse

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_leafs)
        # if self.padding_idx is not None:
        #     with torch.no_grad():
        #         self.weight_leafs[self.padding_idx].fill_(0)

    def get_weights(self):
        w = self.weight_leafs

        # TODO: grab the input from the w and perform all the operations just on that
        # w = w[:,:,input_1d,:]  #DEVICE ASSERTION ERROR WHEN DOING MSE !?!
        # w = nn.LayerNorm(w.shape[-1]).cuda()(w) # not used in experiments before.
        if self.order == 2:
            w01 = (w[0, :, :, :, None] * w[1, :, :, None, :])
            w01 = w01.view(self.rank, self.num_embeddings, -1)
            # w01 = nn.LayerNorm(w01.shape[-2:]).cuda()(w01)

            weight = w01.sum(0)
        elif self.order == 4:
            w01 = (w[0, :, :, :, None] * w[1, :, :, None, :])
            w01 = w01.view(self.rank, self.num_embeddings, -1)
            # w01 = nn.LayerNorm(w01.shape[-2:]).cuda()(w01)

            w23 = (w[2, :, :, :, None] * w[3, :, :, None, :])
            w23 = w23.view(self.rank, self.num_embeddings, -1)
            # w23 = nn.LayerNorm(w23.shape[-2:]).cuda()(w23)

            w0123 = (w01[:, :, :, None] * w23[:, :, None, :])
            w0123 = w0123.view(self.rank, self.num_embeddings, -1)
            # w0123 = nn.LayerNorm(w0123.shape[-2:]).cuda()(w0123)

            weight = w0123.sum(0)
        elif self.order == 8:
            w01 = (w[0, :, :, :, None] * w[1, :, :, None, :])
            w01 = w01.view(self.rank, self.num_embeddings, -1)
            w23 = (w[2, :, :, :, None] * w[3, :, :, None, :])
            w23 = w23.view(self.rank, self.num_embeddings, -1)
            w45 = (w[4, :, :, :, None] * w[5, :, :, None, :])
            w45 = w45.view(self.rank, self.num_embeddings, -1)
            w67 = (w[6, :, :, :, None] * w[7, :, :, None, :])
            w67 = w67.view(self.rank, self.num_embeddings, -1)
            w0123 = (w01[:, :, :, None] * w23[:, :, None, :])
            w0123 = w0123.view(self.rank, self.num_embeddings, -1)
            w4567 = (w45[:, :, :, None] * w67[:, :, None, :])
            w4567 = w4567.view(self.rank, self.num_embeddings, -1)
            w01234567 = (w0123[:, :, :, None] * w4567[:, :, None, :])
            w01234567 = w01234567.view(self.rank, self.num_embeddings, -1)
            weight = w01234567.sum(0)
        else:
            raise Exception(f'The order {self.order} is not yet implemented')

        weight = weight[:, :self.embedding_dim]
        return weight

    def forward(self, input):

        weight = self.get_weights()
        # input_1d = input
        # if input.dim() == 2:
        #     input_1d = input.contiguous().view(1, -1)

        return F.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def knocker_product(self, a, b):
        res = []
        for i in range(a.size(-2)):
            row_res = []
            for j in range(a.size(-1)):
                row_res.append(b * a[..., i, j].unsqueeze(-1).unsqueeze(-2))
            res.append(torch.cat(row_res, -1))
        return torch.cat(res, -2)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False):
        assert embeddings.dim() == 4, \
            'Embeddings parameter is expected to be 4-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        embedding.weight.requires_grad = not freeze
        return embedding


class EmbeddingKetXS(BaseEmbedding):
    r"""This is a new embedding using Kronecker products.
    """
    __constants__ = ['num_embeddings', 'embedding_dim', 'order', 'rank', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    def __init__(self, num_embeddings, embedding_dim, order=4, rank=1, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None, lazy=False):
        """
            order: number of times we do the tensor product
            rank : Rank of the matrix, the dimension that we calcualte the sum of batches
        """
        super(EmbeddingKetXS, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_embeddings_leaf = math.ceil((num_embeddings) ** (1 / order))
        self.embedding_dim_leaf = math.ceil((embedding_dim) ** (1 / order))
        logging.info('EmbeddingKetXS num_embeddings_leaf: ' + str(self.num_embeddings_leaf))
        logging.info('EmbeddingKetXS embedding_dim_leaf: ' + str(self.embedding_dim_leaf))
        self.rank = rank
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.order = order
        if _weight is None:
            # Creating Leaf Weights for Kronecker product
            self.weight_leafs = nn.Parameter(torch.Tensor(
                self.order, self.rank, self.num_embeddings_leaf, self.embedding_dim_leaf))
            logging.info('EmbeddingKetXS weight_leafs shape: ' + str(self.weight_leafs.shape))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [self.order, self.rank, self.num_embeddings_leaf, self.embedding_dim_leaf], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight_leafs = nn.Parameter(_weight)
        self.sparse = sparse
        self.lazy = lazy

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight_leafs)
        torch.nn.init.normal_(self.weight_leafs)
        # if self.padding_idx is not None:  # I can't set the value of a specific location to zero since I don't know it's final location.
        #     with torch.no_grad():
        #         self.weight_leafs[self.padding_idx].fill_(0)

    def get_weights(self):
        w = self.weight_leafs
        weight_leafs_product = w[0]
        for i in range(1, self.order):
            weight_leafs_product = self.knocker_product(weight_leafs_product, w[i])

        weight = weight_leafs_product.sum(dim=0)
        weight = weight[:self.num_embeddings, :self.embedding_dim]
        return weight

    def forward(self, input):
        # Here I should calculate the (final) weight first using tensor products and the rest is exactly the same
        # w = nn.BatchNorm2d(self.weight_leafs.shape[1]).cuda()(self.weight_leafs)

        if self.lazy:
            w = self.weight_leafs

            self.weight = KroneckerProductLazyTensor(*NonLazyTensor(w)).sum(
                dim=0)  # get the sum of the batch of product
            logging.debug('self.weight.shape: ' + str(self.weight.shape))
            if input.dim() == 1:  #
                return self.weight[input].base_lazy_tensor.evaluate().sum(dim=-3)[:,
                       :self.embedding_dim]  # https://github.com/cornellius-gp/gpytorch/pull/871
            elif input.dim() == 2:
                input_1d = input.contiguous().view(1, -1)
                result = self.weight[input_1d[0]].base_lazy_tensor.evaluate().sum(dim=-3)[:,
                         :self.embedding_dim]  # TODO: Not sure if this selection (self.embedding_dim) is correct in here. # https://github.com/cornellius-gp/gpytorch/pull/871
                return result.view(input.shape[0], input.shape[1], -1)
            else:
                raise Exception('This input dimesion is not yet implemented')
        else:

            self.weight = self.get_weights()

            return F.embedding(
                input, self.weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)

    def knocker_product(self, a, b):
        res = []
        for i in range(a.size(-2)):
            row_res = []
            for j in range(a.size(-1)):
                row_res.append(b * a[..., i, j].unsqueeze(-1).unsqueeze(-2))
            res.append(torch.cat(row_res, -1))
        return torch.cat(res, -2)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False):
        assert embeddings.dim() == 4, \
            'Embeddings parameter is expected to be 4-dimensional'
        order, rank, rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        embedding.weight.requires_grad = not freeze
        return embedding

INPUT_DIM,EMBEDDING_DIM = 30522,768

def getTTEmbedding(INPUT_DIM,EMBEDDING_DIM,embedding_type = "tt",d=3,rank = 8,padding_idx = 0): # tr
    import t3nsor as t3
    if embedding_type == "tt":
        embed_model = t3.TTEmbedding(
            voc_size=INPUT_DIM,
            emb_size=EMBEDDING_DIM,
            auto_shapes=True,
            auto_shape_mode='mixed',
            d=d,
            tt_rank=rank,
            padding_idx=padding_idx
        )
        compression_rate = INPUT_DIM * EMBEDDING_DIM / embed_model.tt_matrix.dof
    else:
        embed_model = t3.TREmbedding(
            voc_size=INPUT_DIM,
            emb_size=EMBEDDING_DIM,
            auto_shapes=True,
            auto_shape_mode='mixed',
            d=d,
            tt_rank=rank,
            padding_idx=padding_idx
        )
        compression_rate = INPUT_DIM * EMBEDDING_DIM / embed_model.tr_matrix.dof
    print(compression_rate)
    return embed_model
def test_time(embebding_layer,order, rank):
    model = embebding_layer(30522, 768, rank=rank, order=order)  # XS
    scaled = sum([reduce(lambda x, y: x * y, i.size()) for i in model.parameters()])
    print("ori: {} : {}  with compression rate {}".format(30522 * 768, scaled, 30522 * 768 / scaled))
    # print(model.summary())
    # embeddings = super(nn.Embedding, self).__init__()
    # embeddings = np.random.rand(30522,768)
    # embeddings_layer.initialize(embeddings)
    # model = torch.load("D:\\models\\bert-base-uncased\\pytorch_model.bin")

    # embeddings = model["bert.embeddings.word_embeddings.weight"]
    # # embeddings = np.random.rand(30522,768)
    # print(embeddings)
    # embeddings_layer.initialize(embeddings)
    # print(embeddings)
    print(model.get_weights().shape)
    x = [i for i in range(32)]
    x = torch.LongTensor(x).view(2, 16)
    print(x)
    # embeddings_layer(x)
    for i in range(2):
        start = time.time()
        y = model(x)
        end = time.time() - start
        print("time consumed {}".format(end))
    return end

def main():
    import numpy as np
    for embebding_layer in [EmbeddingKet,EmbeddingKetXS]:
        for order in [2,4,8]:
            for rank in range(1,10,1):
                print(embebding_layer,order, rank)
                test_time(embebding_layer,order,rank)

if __name__ == "__main__":
    main()
