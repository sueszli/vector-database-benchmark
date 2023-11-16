from __future__ import division
import numpy as np
from neon import NervanaObject

class DataLoaderTransformer(NervanaObject):
    """
    DataLoaderTransformers are used to transform the output of a DataLoader.
    DataLoader doesn't have easy access to the device or graph, so any
    computation that should happen there should use a DataLoaderTransformer.
    """

    def __init__(self, dataloader, index=None):
        if False:
            while True:
                i = 10
        super(DataLoaderTransformer, self).__init__()
        self.dataloader = dataloader
        self.index = index
        if self.index is not None:
            data_size = np.prod(self.dataloader.shapes()[index])
            self._shape = (data_size, self.be.bsz)

    def __getattr__(self, key):
        if False:
            while True:
                i = 10
        return getattr(self.dataloader, key)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        for tup in self.dataloader:
            if self.index is None:
                yield self.transform(tup)
            else:
                ret = self.transform(tup[self.index])
                if ret is None:
                    raise ValueError('{} returned None from a transformer'.format(self.__class__.__name__))
                out = list(tup)
                out[self.index] = ret
                yield out

    def transform(self, t):
        if False:
            i = 10
            return i + 15
        raise NotImplemented()

class OneHot(DataLoaderTransformer):
    """
    OneHot will convert `index` into a onehot vector.
    """

    def __init__(self, dataloader, index, nclasses, *args, **kwargs):
        if False:
            print('Hello World!')
        super(OneHot, self).__init__(dataloader, index, *args, **kwargs)
        self.output = self.be.iobuf(nclasses, parallelism='Data')

    def transform(self, t):
        if False:
            print('Hello World!')
        self.output[:] = self.be.onehot(t, axis=0)
        return self.output

class PixelWiseOneHot(DataLoaderTransformer):
    """
    OneHot will convert `index` into a onehot vector.
    """

    def __init__(self, dataloader, index, nclasses, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(PixelWiseOneHot, self).__init__(dataloader, index, *args, **kwargs)
        self.output = None
        self.nclasses = nclasses

    def transform(self, t):
        if False:
            while True:
                i = 10
        if self.output is None:
            self.output = self.be.iobuf(self.nclasses * t.shape[0], dtype=np.int32)
            self.outview = self.output.reshape((self.nclasses, -1))
        self.outview[:] = self.be.onehot(t.reshape((1, -1)), axis=0)
        return self.output

class TypeCast(DataLoaderTransformer):
    """
    TypeCast data from dataloader at `index` to dtype and move into
    device memory if not already.
    """

    def __init__(self, dataloader, index, dtype, *args, **kwargs):
        if False:
            print('Hello World!')
        super(TypeCast, self).__init__(dataloader, *args, index=index, **kwargs)
        self.output = self.be.iobuf(self._shape[0], dtype=dtype, parallelism='Data')

    def transform(self, t):
        if False:
            print('Hello World!')
        self.output[:] = t
        return self.output

class Retuple(DataLoaderTransformer):
    """
    Converts data from dataloader to a tuple of tuples with first element as
    the input tuple and second element as the target tuple.
    """

    def __init__(self, dataloader, data=(0,), target=(1,), *args, **kwargs):
        if False:
            print('Hello World!')
        super(Retuple, self).__init__(dataloader, *args, index=None, **kwargs)
        self._data = data
        self._target = target
        self.output = None

    def transform(self, t):
        if False:
            return 10
        if len(self._data) > 1:
            data = tuple((t[ii] for ii in self._data))
        else:
            data = t[self._data[0]]
        if len(self._target) > 1:
            target = tuple((t[ii] for ii in self._target))
        else:
            target = t[self._target[0]]
        return (data, target)

class BGRMeanSubtract(DataLoaderTransformer):
    """
    subtract pixel_mean from data at `index`.  Assumes data is in CxHxWxN
    """

    def __init__(self, dataloader, index, pixel_mean=[127, 119, 104], *args, **kwargs):
        if False:
            print('Hello World!')
        super(BGRMeanSubtract, self).__init__(dataloader, *args, index=index, **kwargs)
        pixel_mean = np.asarray(pixel_mean)
        self.pixel_mean = self.be.array(pixel_mean[:, np.newaxis])

    def transform(self, t):
        if False:
            print('Hello World!')
        tr = t.reshape((3, -1))
        tr[:] = tr - self.pixel_mean
        return t

class ValueNormalize(DataLoaderTransformer):
    """
    normalize values at `index`
    """

    def __init__(self, dataloader, index, source_range=[0.0, 255.0], target_range=[0.0, 1.0], *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(ValueNormalize, self).__init__(dataloader, *args, index=index, **kwargs)
        source_range = np.asarray(source_range)
        target_range = np.asarray(target_range)
        self.xmin = self.be.array(source_range[0])
        self.xspan = self.be.array(source_range[1] - source_range[0])
        self.ymin = self.be.array(target_range[0])
        self.yspan = self.be.array(target_range[1] - target_range[0])

    def transform(self, t):
        if False:
            for i in range(10):
                print('nop')
        tr = t.reshape((3, -1))
        tr[:] = (tr - self.xmin) / self.xspan * self.yspan + self.ymin
        return t

class DumpImage(DataLoaderTransformer):

    def __init__(self, dataloader, index, image_index, outshape, output_directory=None, *args, **kwargs):
        if False:
            return 10
        '\n        dump image number `image_index` in data `index` to a random\n        file in `output_directory`.\n        '
        super(DumpImage, self).__init__(dataloader, *args, index=index, **kwargs)
        self.outshape = outshape
        self.image_index = image_index
        self.output_directory = output_directory or '/tmp'
        if self.output_directory[-1] != '/':
            self.output_directory += '/'

    def transform(self, t):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(t, np.ndarray):
            a = t
        else:
            a = t.get()
        a = a[:, self.image_index]
        if self.outshape[0] is 1:
            nshape = (3, self.outshape[1], self.outshape[2])
            img2 = np.ndarray(nshape, dtype='uint8')
            a = a.reshape((self.outshape[1], self.outshape[2]))
            img2[0, :, :] = a
            img2[1, :, :] = a
            img2[2, :, :] = a
            a = img2
        else:
            a = a.reshape(self.outshape)
        a = a.transpose(1, 2, 0)
        a = a[:, :, ::-1]
        a = a.astype('uint8')
        from PIL import Image as PILImage
        img = PILImage.fromarray(a)
        img.save(self.filename())
        return t

    def filename(self):
        if False:
            return 10
        '\n        generate random filename\n        '
        import random
        return self.output_directory + str(random.random()) + '.png'