class Serializable(object):
    """Serializable from and to JSON with same mechanism as Keras Layer."""

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the current config of this object.\n\n        # Returns\n            Dictionary.\n        '
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        if False:
            while True:
                i = 10
        'Build an instance from the config of this object.\n\n        # Arguments\n            config: Dict. The config of the object.\n        '
        return cls(**config)