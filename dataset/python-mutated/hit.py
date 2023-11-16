from ..utils import AttrDict, HitMeta

class Hit(AttrDict):

    def __init__(self, document):
        if False:
            for i in range(10):
                print('nop')
        data = {}
        if '_source' in document:
            data = document['_source']
        if 'fields' in document:
            data.update(document['fields'])
        super().__init__(data)
        super(AttrDict, self).__setattr__('meta', HitMeta(document))

    def __getstate__(self):
        if False:
            print('Hello World!')
        return super().__getstate__() + (self.meta,)

    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        super(AttrDict, self).__setattr__('meta', state[-1])
        super().__setstate__(state[:-1])

    def __dir__(self):
        if False:
            print('Hello World!')
        return super().__dir__() + ['meta']

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<Hit({}): {}>'.format('/'.join((getattr(self.meta, key) for key in ('index', 'id') if key in self.meta)), super().__repr__())