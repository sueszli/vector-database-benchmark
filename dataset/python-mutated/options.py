from factory import declarations
from factory.base import FactoryOptions, OptionDefault

class BlockFactoryOptions(FactoryOptions):

    def _build_default_options(self):
        if False:
            print('Hello World!')
        options = super()._build_default_options()
        options.append(OptionDefault('block_def', None))
        return options

    def get_meta_dict(self):
        if False:
            return 10
        return {'model': self.model, 'block_def': self.block_def, 'abstract': self.abstract, 'strategy': self.strategy, 'inline_args': self.inline_args, 'exclude': self.exclude, 'rename': self.rename}

    def to_meta_class(self):
        if False:
            print('Hello World!')
        "\n        Create a new Meta class from this instance's options, suitable for\n        inclusion on a factory subclass\n        "
        return type('Meta', (), self.get_meta_dict())

class StreamBlockFactoryOptions(BlockFactoryOptions):

    def prepare_arguments(self, attributes):
        if False:
            for i in range(10):
                print('nop')

        def get_block_name(key):
            if False:
                for i in range(10):
                    print('nop')
            return key.split('.')[1]
        kwargs = dict(attributes)
        kwargs = self.factory._adjust_kwargs(**kwargs)
        filtered_kwargs = {}
        for (k, v) in kwargs.items():
            block_name = get_block_name(k)
            if block_name not in self.exclude and block_name not in self.parameters and (v is not declarations.SKIP):
                filtered_kwargs[k] = v
        return ((), filtered_kwargs)

    def get_block_definition(self):
        if False:
            print('Hello World!')
        if self.block_def is not None:
            return self.block_def
        elif self.model is not None:
            return self.model()