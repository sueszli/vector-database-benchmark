import torch
from tokenizers import Tokenizer
from transformers import StoppingCriteria

class SequenceStoppingCriteria(StoppingCriteria):
    """Enables automatic stopping of model text generation when specific text sequences are generated."""

    def __init__(self, tokenizer: Tokenizer, stop_texts: list[str], input_prompt: str, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.stop_texts = stop_texts
        self.tokenizer = tokenizer
        self.input_length = len(tokenizer.encode(input_prompt))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if False:
            print('Hello World!')
        generated_ids = input_ids[0, self.input_length:].tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        return any((text in generated_text for text in self.stop_texts))