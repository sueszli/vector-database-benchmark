from typing import List, Optional
import logging
from allennlp.common import Registrable
from allennlp.data.tokenizers.token_class import Token
logger = logging.getLogger(__name__)

class Tokenizer(Registrable):
    """
    A `Tokenizer` splits strings of text into tokens.  Typically, this either splits text into
    word tokens or character tokens, and those are the two tokenizer subclasses we have implemented
    here, though you could imagine wanting to do other kinds of tokenization for structured or
    other inputs.

    See the parameters to, e.g., :class:`~.SpacyTokenizer`, or whichever tokenizer
    you want to use.

    If the base input to your model is words, you should use a :class:`~.SpacyTokenizer`, even if
    you also want to have a character-level encoder to get an additional vector for each word
    token.  Splitting word tokens into character arrays is handled separately, in the
    :class:`..token_representations.TokenRepresentation` class.
    """
    default_implementation = 'spacy'

    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        if False:
            print('Hello World!')
        '\n        Batches together tokenization of several texts, in case that is faster for particular\n        tokenizers.\n\n        By default we just do this without batching.  Override this in your tokenizer if you have a\n        good way of doing batched computation.\n        '
        return [self.tokenize(text) for text in texts]

    def tokenize(self, text: str) -> List[Token]:
        if False:
            while True:
                i = 10
        '\n        Actually implements splitting words into tokens.\n\n        # Returns\n\n        tokens : `List[Token]`\n        '
        raise NotImplementedError

    def add_special_tokens(self, tokens1: List[Token], tokens2: Optional[List[Token]]=None) -> List[Token]:
        if False:
            return 10
        '\n        Adds special tokens to tokenized text. These are tokens like [CLS] or [SEP].\n\n        Not all tokenizers do this. The default is to just return the tokens unchanged.\n\n        # Parameters\n\n        tokens1 : `List[Token]`\n            The list of tokens to add special tokens to.\n        tokens2 : `Optional[List[Token]]`\n            An optional second list of tokens. This will be concatenated with `tokens1`. Special tokens will be\n            added as appropriate.\n\n        # Returns\n        tokens : `List[Token]`\n            The combined list of tokens, with special tokens added.\n        '
        return tokens1 + (tokens2 or [])

    def num_special_tokens_for_sequence(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Returns the number of special tokens added for a single sequence.\n        '
        return 0

    def num_special_tokens_for_pair(self) -> int:
        if False:
            print('Hello World!')
        '\n        Returns the number of special tokens added for a pair of sequences.\n        '
        return 0