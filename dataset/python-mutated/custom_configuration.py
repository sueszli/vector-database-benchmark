from transformers import PretrainedConfig

class CustomConfig(PretrainedConfig):
    model_type = 'custom'

    def __init__(self, attribute=1, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.attribute = attribute
        super().__init__(**kwargs)

class NoSuperInitConfig(PretrainedConfig):
    model_type = 'custom'

    def __init__(self, attribute=1, **kwargs):
        if False:
            i = 10
            return i + 15
        self.attribute = attribute