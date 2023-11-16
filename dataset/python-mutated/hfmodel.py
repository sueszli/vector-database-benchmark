"""
Hugging Face Transformers model wrapper module
"""
from ..models import Models
from .tensors import Tensors

class HFModel(Tensors):
    """
    Pipeline backed by a Hugging Face Transformers model.
    """

    def __init__(self, path=None, quantize=False, gpu=False, batch=64):
        if False:
            while True:
                i = 10
        '\n        Creates a new HFModel.\n\n        Args:\n            path: optional path to model, accepts Hugging Face model hub id or local path,\n                  uses default model for task if not provided\n            quantize: if model should be quantized, defaults to False\n            gpu: True/False if GPU should be enabled, also supports a GPU device id\n            batch: batch size used to incrementally process content\n        '
        self.path = path
        self.quantization = quantize
        self.deviceid = Models.deviceid(gpu)
        self.device = Models.device(self.deviceid)
        self.batchsize = batch

    def prepare(self, model):
        if False:
            print('Hello World!')
        '\n        Prepares a model for processing. Applies dynamic quantization if necessary.\n\n        Args:\n            model: input model\n\n        Returns:\n            model\n        '
        if self.deviceid == -1 and self.quantization:
            model = self.quantize(model)
        return model

    def tokenize(self, tokenizer, texts):
        if False:
            i = 10
            return i + 15
        '\n        Tokenizes text using tokenizer. This method handles overflowing tokens and automatically splits\n        them into separate elements. Indices of each element is returned to allow reconstructing the\n        transformed elements after running through the model.\n\n        Args:\n            tokenizer: Tokenizer\n            texts: list of text\n\n        Returns:\n            (tokenization result, indices)\n        '
        (batch, positions) = ([], [])
        for (x, text) in enumerate(texts):
            elements = [t + ' ' for t in text.split('\n') if t]
            batch.extend(elements)
            positions.extend([x] * len(elements))
        tokens = tokenizer(batch, padding=True)
        (inputids, attention, indices) = ([], [], [])
        for (x, ids) in enumerate(tokens['input_ids']):
            if len(ids) > tokenizer.model_max_length:
                ids = [i for i in ids if i != tokenizer.pad_token_id]
                for chunk in self.batch(ids, tokenizer.model_max_length - 1):
                    if chunk[-1] != tokenizer.eos_token_id:
                        chunk.append(tokenizer.eos_token_id)
                    mask = [1] * len(chunk)
                    if len(chunk) < tokenizer.model_max_length:
                        pad = tokenizer.model_max_length - len(chunk)
                        chunk.extend([tokenizer.pad_token_id] * pad)
                        mask.extend([0] * pad)
                    inputids.append(chunk)
                    attention.append(mask)
                    indices.append(positions[x])
            else:
                inputids.append(ids)
                attention.append(tokens['attention_mask'][x])
                indices.append(positions[x])
        tokens = {'input_ids': inputids, 'attention_mask': attention}
        return ({name: self.tensor(tensor).to(self.device) for (name, tensor) in tokens.items()}, indices)