"""Entropy coder model."""

class EntropyCoderModel(object):
    """Entropy coder model."""

    def __init__(self):
        if False:
            return 10
        self.loss = None
        self.train_op = None
        self.average_code_length = None

    def Initialize(self, global_step, optimizer, config_string):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def BuildGraph(self, input_codes):
        if False:
            i = 10
            return i + 15
        'Build the Tensorflow graph corresponding to the entropy coder model.\n\n    Args:\n      input_codes: Tensor of size: batch_size x height x width x bit_depth\n        corresponding to the codes to compress.\n        The input codes are {-1, +1} codes.\n    '
        raise NotImplementedError()

    def GetConfigStringForUnitTest(self):
        if False:
            return 10
        'Returns a default model configuration to be used for unit tests.'
        return None