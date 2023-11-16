"""
Texts module
"""
from itertools import chain
from .base import Data

class Texts(Data):
    """
    Tokenizes text datasets as input for training language models.
    """

    def __init__(self, tokenizer, columns, maxlength):
        if False:
            return 10
        '\n        Creates a new instance for tokenizing Texts training data.\n\n        Args:\n            tokenizer: model tokenizer\n            columns: tuple of columns to use for text\n            maxlength: maximum sequence length\n        '
        super().__init__(tokenizer, columns, maxlength)
        if not self.columns:
            self.columns = ('text', None)

    def process(self, data):
        if False:
            return 10
        (text1, text2) = self.columns
        text = (data[text1], data[text2]) if text2 else (data[text1],)
        inputs = self.tokenizer(*text, return_special_tokens_mask=True)
        return self.concat(inputs)

    def concat(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Concatenates tokenized text into chunks of maxlength.\n\n        Args:\n            inputs: tokenized input\n\n        Returns:\n            Chunks of tokenized text each with a size of maxlength\n        '
        concat = {k: list(chain(*inputs[k])) for k in inputs.keys()}
        length = len(concat[list(inputs.keys())[0]])
        if length >= self.maxlength:
            length = length // self.maxlength * self.maxlength
        result = {k: [v[x:x + self.maxlength] for x in range(0, length, self.maxlength)] for (k, v) in concat.items()}
        return result