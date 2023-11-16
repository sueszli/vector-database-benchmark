class Analyser(object):
    """Analyze the dataset for useful information.

    Analyser is used by the input nodes and the heads of the hypermodel.  It
    analyzes the dataset to get useful information, e.g., the shape of the
    data, the data type of the dataset. The information will be used by the
    input nodes and heads to construct the data pipeline and to build the Keras
    Model.
    """

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.shape = None
        self.dtype = None
        self.num_samples = 0
        self.batch_size = None

    def update(self, data):
        if False:
            return 10
        'Update the statistics with a batch of data.\n\n        # Arguments\n            data: tf.Tensor. One batch of data from tf.data.Dataset.\n        '
        if self.dtype is None:
            self.dtype = data.dtype
        if self.shape is None:
            self.shape = data.shape.as_list()
        if self.batch_size is None:
            self.batch_size = data.shape.as_list()[0]
        self.num_samples += data.shape.as_list()[0]

    def finalize(self):
        if False:
            return 10
        'Process recorded information after all updates.'
        raise NotImplementedError