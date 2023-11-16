import abc

class SamplingTrainableMixin(metaclass=abc.ABCMeta):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self._train_param_blobs = None
        self._train_param_blobs_frozen = False

    @property
    @abc.abstractmethod
    def param_blobs(self):
        if False:
            while True:
                i = 10
        '\n        List of parameter blobs for prediction net\n        '
        pass

    @property
    def train_param_blobs(self):
        if False:
            i = 10
            return i + 15
        '\n        If train_param_blobs is not set before used, default to param_blobs\n        '
        if self._train_param_blobs is None:
            self.train_param_blobs = self.param_blobs
        return self._train_param_blobs

    @train_param_blobs.setter
    def train_param_blobs(self, blobs):
        if False:
            for i in range(10):
                print('nop')
        assert not self._train_param_blobs_frozen
        assert blobs is not None
        self._train_param_blobs_frozen = True
        self._train_param_blobs = blobs

    @abc.abstractmethod
    def _add_ops(self, net, param_blobs):
        if False:
            return 10
        '\n        Add ops to the given net, using the given param_blobs\n        '
        pass

    def add_ops(self, net):
        if False:
            for i in range(10):
                print('nop')
        self._add_ops(net, self.param_blobs)

    def add_train_ops(self, net):
        if False:
            return 10
        self._add_ops(net, self.train_param_blobs)