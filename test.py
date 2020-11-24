import numpy as np
from glob import glob
import os
import torch
import torch.nn as nn
import torch
import time
import datetime

def one():
    link = 'C:/Users/yinan/Desktop/data/old_feature/Commits/commit/Commits_Graph/'

    number = [959]
    other = 'C:/Users/yinan/Desktop/data/old_feature/GoogleDoc/Very_Suspicious/Commits_Graph/10.npz'

    data_list_susp = glob('C:/Users/yinan/Desktop/data/old_feature/Diffs_old_result/*/Commits_Graph/*')

    for num in number:
        path = link + str(num) + '.npz'
        n = np.load(path)
        of = n['graph_node_old']
        nf = n['graph_node_new']
        print(num)
        print('new')
        #print(nf[:, :3])
        print(np.sum(nf, axis=1))
        print(np.sum((nf != 0), axis=1))
        print('old')
        #print(of[:, :3])
        print(np.sum(of, axis=1))
        print(np.sum((of != 0), axis=1))
        print('')

    # n = np.load(other)
    # of = n['graph_node_old']
    # nf = n['graph_node_new']
    # print(10)
    # print('new')
    # print(nf[:, :3])
    # print(np.sum(nf, axis=1))
    # print(np.sum((nf != 0), axis=1))
    # print('old')
    # print(of[:, :3])
    # print(np.sum(of, axis=1))
    # print(np.sum((of != 0), axis=1))
    # print('')


def two():
    link = 'C:/Users/yinan/Desktop/data/old_feature/Commits/commit/Commits_Graph/'

    number = [1661, 181, 1670, 1665, 1812, 1736, 1804, 1665, 1797, 1729, 1801, 1670, 1542,
              181]
    other = 'C:/Users/yinan/Desktop/data/old_feature/GoogleDoc/Very_Suspicious/Commits_Graph/10.npz'
    print(len(set(number)))

    for num in number:
        path = link + str(num) + '.npz'
        n = np.load(path)
        of = n['graph_node_old']
        nf = n['graph_node_new']
        oe = n['graph_edge_old']
        ne = n['graph_edge_new']
        print(num)
        print('new')
        print(ne)
        print('old')
        print(oe)
        print('')


def three():
    data_list_new = glob('C:/Users/yinan/Desktop/data/old_feature/Commits/commit/Commits_Graph/*')
    data_list_susp = glob('C:/Users/yinan/Desktop/data/old_feature/GoogleDoc/Very_Suspicious/Commits_Graph/*')
    data_list_draft = glob('C:/Users/yinan/Desktop/data/old_feature/GoogleDoc/Commits_Draft/Commits_Graph/*')

    ft_list_bad = data_list_susp.copy()
    ft_list_bad.append(data_list_draft[-1])
    test_list = data_list_new[150:350] + ft_list_bad

    result = np.zeros(1,dtype=float)

    for i in test_list:
        n = np.load(i)
        of = n['graph_node_old']
        nf = n['graph_node_new']

        r = np.sum(np.abs(nf - of))[np.newaxis,]

        result = np.concatenate((result, r), axis=0)

    dissimilarity = torch.from_numpy(result[1:])

    _, indicates = torch.topk(dissimilarity, k=15, dim=-1)
    print("Dissimilarity Top K")
    for i in indicates:
        print("{}:  {}".format(test_list[i], dissimilarity[i]))
    print("Finish! \n")


class mte(torch.Tensor):
    def __init__(self, tensor):
        self.m = 0
        self.k =tensor-3
    def new_softmax(self,dim,p):
        if self.dim() == 0:
            assert dim == 0, "Improper dim argument"
            return MPCTensor(torch.ones(()))

        if self.size(dim) == 1:
            return MPCTensor(torch.ones(self.size()))

        maximum_value = self.max(dim, keepdim=True)[0]
        logits = self - maximum_value
        numerator = logits.exp()
        inv_denominator = numerator.sum(dim, keepdim=True)
        return numerator / inv_denominator

    def pow(self, p, **kwargs):
        """
        Computes an element-wise exponent `p` of a tensor, where `p` is an
        integer.
        """
        if isinstance(p, float) and int(p) == p:
            p = int(p)

        if not isinstance(p, int):
            raise TypeError(
                "pow must take an integer exponent. For non-integer powers, use"
                " pos_pow with positive-valued base."
            )
        if p < -1:
            return self.reciprocal(**kwargs).pow(-p)
        elif p == -1:
            return self.reciprocal(**kwargs)
        elif p == 0:
            # Note: This returns 0 ** 0 -> 1 when inputs have zeros.
            # This is consistent with PyTorch's pow function.
            return MPCTensor(torch.ones(self.size()))
        elif p == 1:
            return self.clone()
        elif p == 2:
            return self.square()
        elif p % 2 == 0:
            return self.square().pow(p // 2)
        else:
            return self.square().mul_(self).pow((p - 1) // 2)


class function(torch.nn.Module):
    def __init__(self):
        super(function, self).__init__()

    def forward(self, logits):
        inv = torch.sum(torch.pow(logits,2), dim=-1, keepdim=True)
        return torch.log(torch.pow(logits,2)/inv), 1/inv

def _is_type_tensor(tensor, types):
    """Checks whether the elements of the input tensor are of a given type"""
    if torch.is_tensor(tensor):
        if any(tensor.dtype == type_ for type_ in types):
            return True
    return False


def is_float_tensor(tensor):
    """Checks if the input tensor is a Torch tensor of a float type."""
    return _is_type_tensor(tensor, [torch.float16, torch.float32, torch.float64])


def is_int_tensor(tensor):
    """Checks if the input tensor is a Torch tensor of an int type."""
    return _is_type_tensor(
        tensor, [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
    )

class FixedPointEncoder:
    """Encoder that encodes long or float tensors into scaled integer tensors."""

    __default_precision_bits = 16

    def __init__(self, precision_bits=None):
        if precision_bits is None:
            precision_bits = FixedPointEncoder.__default_precision_bits
        self._scale = int(2 ** precision_bits)

    def encode(self, x):
        """Helper function to wrap data if needed"""
        if isinstance(x, int) or isinstance(x, float):
            # Squeeze in order to get a 0-dim tensor with value `x`
            return torch.LongTensor([self._scale * x]).squeeze()
        elif isinstance(x, list):
            return torch.FloatTensor(x).mul_(self._scale).long()
        elif is_float_tensor(x):
            return (self._scale * x).long()
        # For integer types cast to long prior to scaling to avoid overflow.
        elif is_int_tensor(x):
            return self._scale * x.long()
        elif isinstance(x, np.ndarray):
            return self._scale * torch.from_numpy(x).long()
        elif torch.is_tensor(x):
            raise TypeError("Cannot encode input with dtype %s" % x.dtype)
        else:
            raise TypeError("Unknown tensor type: %s." % type(x))

    def decode(self, tensor):
        """Helper function that decodes from scaled tensor"""
        if tensor is None:
            return None
        assert is_int_tensor(tensor), "input must be a LongTensor"
        if self._scale > 1:
            correction = (tensor < 0).long()
            dividend = tensor / self._scale - correction
            remainder = tensor % self._scale
            remainder += (remainder == 0).long() * self._scale * correction

            tensor = dividend.float() + remainder.float() / self._scale
        else:
            tensor = nearest_integer_division(tensor, self._scale)

        return tensor

    @property
    def scale(self):
        return self._scale

    @classmethod
    def set_default_precision(cls, precision_bits):
        assert (
            isinstance(precision_bits, int)
            and precision_bits >= 0
            and precision_bits < 64
        ), "precision must be a positive integer less than 64"
        cls.__default_precision_bits = precision_bits

class encoder(nn.Module):
    def __init__(self, hbead=6):
        super(encoder, self).__init__()
        layers = [nn.Linear(64,64) for _ in range(hbead)]
        self.layer = nn.Sequential(*layers)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, x, i=1):
        for k in range(i):
            x = self.layer(x.detach())
        return x

if __name__ == '__main__':
    a = torch.rand((10,1))
    a = a * a.T
    print(a.shape)
    #print(net)





