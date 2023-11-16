from bigdl.orca.tfpark import KerasModel

class TextKerasModel(KerasModel):
    """
    The base class for text models in tfpark.
    """

    def __init__(self, labor, optimizer=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.labor = labor
        self.labor.build(**kwargs)
        model = self.labor.model
        if optimizer:
            model.compile(loss=model.loss, optimizer=optimizer, metrics=model.metrics)
        super(TextKerasModel, self).__init__(model)

    def save_model(self, path):
        if False:
            print('Hello World!')
        '\n        Save the model to a single HDF5 file.\n\n        :param path: String. The path to save the model.\n        '
        self.labor.save(path)

    @staticmethod
    def _load_model(labor, path):
        if False:
            while True:
                i = 10
        labor.load(path)
        model = KerasModel(labor.model)
        model.labor = labor
        return model