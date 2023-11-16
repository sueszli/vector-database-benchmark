from __future__ import division
import numpy as np
from neon.data.dataloader_transformers import DataLoaderTransformer
from collections import OrderedDict

class DataLoaderAdapter(DataLoaderTransformer):
    """
    DataLoaderAdapter converts Aeon data buffers to tensors.

    Arguments:
        dataloader (DataLoader): Aeon dataloading module.
    """

    def __init__(self, dataloader):
        if False:
            print('Hello World!')
        super(DataLoaderAdapter, self).__init__(dataloader, None)
        self.shape = self.shapes()[0]
        (self.nbatches, modal) = divmod(self.dataloader.ndata, self.be.bsz)
        if modal > 0:
            self.nbatches += 1
        self.outputs = OrderedDict()

        def max_dur(val, freq):
            if False:
                print('Hello World!')
            uval = float(val.split(' ')[0])
            ucat = val.split(' ')[1]
            if ucat == 'samples':
                return float(uval)
            elif ucat == 'seconds':
                return float(uval * freq)
            elif ucat == 'milliseconds':
                return float(uval / 1000 * freq)
            else:
                raise ValueError('Unknown time unit ' + ucat)
        for conf in self.dataloader.config['etl']:
            if conf['type'] == 'audio':
                self.max_duration = max_dur(conf['max_duration'], conf['sample_freq_hz'])

    def transform(self, t):
        if False:
            i = 10
            return i + 15
        "\n        Converts Aeon data to tuple of tensors.\n\n        Arguments:\n            t (tuple): Tuple of numpy arrays.\n                For example: {tuple}{'image':ndarray(...), 'label':ndarray(...)}\n                where 'image' shape is (N,C,H,W) and 'label' shape is (N,1)\n        "
        for (key, value) in t:
            assert value.shape[0] == self.be.bsz
            reshape_rows = self.be.bsz
            if key == 'audio_length':
                for x in np.nditer(value, op_flags=['readwrite']):
                    x[...] = x / self.max_duration * 100
                value = value.astype(np.uint8, copy=False)
            if key == 'char_map':
                x = value
                x = x[x != 0]
                x = np.reshape(x, (1, -1))
                x = np.ascontiguousarray(x)
            else:
                x = np.reshape(value, (reshape_rows, -1))
                x = np.ascontiguousarray(x.T)
            x = np.array(x, order='C')
            self.outputs[key] = self.be.array(x, dtype=value.dtype)
        return tuple(self.outputs.values())

    def shapes(self):
        if False:
            return 10
        shapes = []
        for (name, value) in self.dataloader.axes_info:
            vals = ()
            for (child_name, child_value) in value:
                vals = vals + (child_value,)
            shapes.append(vals)
        return tuple(shapes)