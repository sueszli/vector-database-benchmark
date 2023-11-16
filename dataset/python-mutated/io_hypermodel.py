from autokeras.engine import block as block_module

class IOHyperModel(block_module.Block):
    """A mixin class connecting the input nodes and heads with the adapters.

    This class is extended by the input nodes and the heads. The AutoModel calls
    the functions to get the corresponding adapters and pass the information
    back to the input nodes and heads.
    """

    def __init__(self, shape=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.shape = shape
        self.data_shape = None
        self.dtype = None
        self.batch_size = None
        self.num_samples = None

    def get_analyser(self):
        if False:
            print('Hello World!')
        'Get the corresponding Analyser.\n\n        # Returns\n            An instance of a subclass of autokeras.engine.Analyser.\n        '
        raise NotImplementedError

    def get_adapter(self):
        if False:
            i = 10
            return i + 15
        'Get the corresponding Adapter.\n\n        # Returns\n            An instance of a subclass of autokeras.engine.Adapter.\n        '
        raise NotImplementedError

    def config_from_analyser(self, analyser):
        if False:
            print('Hello World!')
        'Load the learned information on dataset from the Analyser.\n\n        # Arguments\n            adapter: An instance of a subclass of autokeras.engine.Adapter.\n        '
        self.data_shape = analyser.shape
        self.dtype = analyser.dtype
        self.batch_size = analyser.batch_size
        self.num_samples = analyser.num_samples

    def get_hyper_preprocessors(self):
        if False:
            while True:
                i = 10
        'Construct a list of HyperPreprocessors based on learned information.\n\n        # Returns\n            A list of HyperPreprocessors for the corresponding data.\n        '
        raise NotImplementedError

    def get_config(self):
        if False:
            return 10
        config = super().get_config()
        config.update({'shape': self.shape})
        return config