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
        self._self_test()

    def _self_test(self):
        shapes = set()
        i = 0
        for normal, simple, w0, w1 in self._data.swapaxes(0, 1):
            shapes.update({normal.shape, simple.shape, w0.shape, w1.shape})
            if len(shapes) > 1:
                raise ValueError('Data inconsistent! (index {})'.format(i))
            i += 1

    def batches(self, batch_size, select=None):
        indices = range(0, self._data.shape[1], batch_size) if not select \
            else np.random.choice(self._data.shape[1], select)
        for b_i, i in enumerate(indices):
            if i + batch_size >= self._data.shape[1]:
                i -= i + batch_size - self._data.shape[1]
            yield b_i, self._data[:, i:i + batch_size]
