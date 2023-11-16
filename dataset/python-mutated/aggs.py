from ..utils import AttrDict, AttrList
from . import AggResponse, Response

class Bucket(AggResponse):

    def __init__(self, aggs, search, data, field=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(aggs, search, data)

class FieldBucket(Bucket):

    def __init__(self, aggs, search, data, field=None):
        if False:
            print('Hello World!')
        if field:
            data['key'] = field.deserialize(data['key'])
        super().__init__(aggs, search, data, field)

class BucketData(AggResponse):
    _bucket_class = Bucket

    def _wrap_bucket(self, data):
        if False:
            return 10
        return self._bucket_class(self._meta['aggs'], self._meta['search'], data, field=self._meta.get('field'))

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self.buckets)

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.buckets)

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        if isinstance(key, (int, slice)):
            return self.buckets[key]
        return super().__getitem__(key)

    @property
    def buckets(self):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(self, '_buckets'):
            field = getattr(self._meta['aggs'], 'field', None)
            if field:
                self._meta['field'] = self._meta['search']._resolve_field(field)
            bs = self._d_['buckets']
            if isinstance(bs, list):
                bs = AttrList(bs, obj_wrapper=self._wrap_bucket)
            else:
                bs = AttrDict({k: self._wrap_bucket(bs[k]) for k in bs})
            super(AttrDict, self).__setattr__('_buckets', bs)
        return self._buckets

class FieldBucketData(BucketData):
    _bucket_class = FieldBucket

class TopHitsData(Response):

    def __init__(self, agg, search, data):
        if False:
            i = 10
            return i + 15
        super(AttrDict, self).__setattr__('meta', AttrDict({'agg': agg, 'search': search}))
        super().__init__(search, data)