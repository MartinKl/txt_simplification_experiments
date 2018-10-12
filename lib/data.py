from functools import reduce
import numpy as np


class DataCollection(dict):
    def __init__(self,
                 normal,
                 simple,
                 normal_weights,
                 simple_weights,
                 splits=(.1, .1),
                 names=('train', 'valid', 'test')):
        super().__init__()
        if sum(splits) >= 1:
            raise ValueError
        split_indices = \
            (len(normal) * np.array(
                reduce(lambda e0, e1: e0 + [(e0[-1][1], e0[-1][1] + e1)], splits, [(0, 1 - sum(splits))])))\
                .astype(np.int32)
        for name, indices in zip(names, split_indices):
            start, end = indices
            self[name] = _DataSet([normal[start:end],
                                   simple[start:end],
                                   normal_weights[start:end],
                                   simple_weights[start:end]])


class _DataSet(object):
    def __init__(self, data):
        self._data = np.array(data)

    def batches(self, batch_size):
        for i in range(0, len(self._data), batch_size):
            yield self._data[:, i:i + batch_size]
