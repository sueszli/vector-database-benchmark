from ..utils import DummyObject, requires_backends
SLOW_TO_FAST_CONVERTERS = None

def convert_slow_tokenizer(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    requires_backends(convert_slow_tokenizer, ['sentencepiece', 'tokenizers'])