from bigdl.nano.tf.keras import Model

class NormalizeTSModel(Model):

    def __init__(self, model, output_feature_dim):
        if False:
            for i in range(10):
                print('nop')
        '\n        Build a Normalization model wrapper.\n        param model: basic forecaster model.\n        :param output_feature_dim: Specify the output dimension.\n        '
        super(NormalizeTSModel, self).__init__()
        self.model = model
        self.output_feature_dim = output_feature_dim

    def call(self, x):
        if False:
            while True:
                i = 10
        seq_last = x[:, -1:, :]
        x = x - seq_last
        y = self.model(x)
        y = y + seq_last[:, :, :self.output_feature_dim]
        return y