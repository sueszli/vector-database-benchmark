"""Wrapper around BigdlLLM embedding models."""
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Extra, Field
from langchain.embeddings.base import Embeddings
DEFAULT_MODEL_NAME = 'gpt2'

class TransformersEmbeddings(BaseModel, Embeddings):
    """Wrapper around bigdl-llm transformers embedding models.

    To use, you should have the ``transformers`` python package installed.

    Example:
        .. code-block:: python

            from bigdl.llm.langchain.embeddings import TransformersEmbeddings
            embeddings = TransformersEmbeddings.from_model_id(model_id)
    """
    model: Any
    'BigDL-LLM Transformers-INT4 model.'
    tokenizer: Any
    'Huggingface tokenizer model.'
    model_id: str = DEFAULT_MODEL_NAME
    'Model name or model path to use.'
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    'Keyword arguments to pass to the model.'
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    'Keyword arguments to pass when calling the `encode` method of the model.'

    @classmethod
    def from_model_id(cls, model_id: str, model_kwargs: Optional[dict]=None, **kwargs: Any):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct object from model_id.\n\n        Args:\n\n            model_id: Path for the huggingface repo id to be downloaded or the huggingface\n                      checkpoint folder.\n            model_kwargs: Keyword arguments that will be passed to the model and tokenizer.\n            kwargs: Extra arguments that will be passed to the model and tokenizer.\n        \n        Returns:\n            An object of TransformersEmbeddings.\n        '
        try:
            from bigdl.llm.transformers import AutoModel
            from transformers import AutoTokenizer, LlamaTokenizer
        except ImportError:
            raise ValueError('Could not import transformers python package. Please install it with `pip install transformers`.')
        _model_kwargs = model_kwargs or {}
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
        except:
            tokenizer = LlamaTokenizer.from_pretrained(model_id, **_model_kwargs)
        model = AutoModel.from_pretrained(model_id, load_in_4bit=True, **_model_kwargs)
        if 'trust_remote_code' in _model_kwargs:
            _model_kwargs = {k: v for (k, v) in _model_kwargs.items() if k != 'trust_remote_code'}
        return cls(model_id=model_id, model=model, tokenizer=tokenizer, model_kwargs=_model_kwargs, **kwargs)

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def embed(self, text: str):
        if False:
            while True:
                i = 10
        'Compute doc embeddings using a HuggingFace transformer model.\n\n        Args:\n            texts: The list of texts to embed.\n\n        Returns:\n            List of embeddings, one for each text.\n        '
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        embeddings = self.model(input_ids, return_dict=False)[0]
        embeddings = embeddings.squeeze(0).detach().numpy()
        embeddings = np.mean(embeddings, axis=0)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if False:
            for i in range(10):
                print('nop')
        'Compute doc embeddings using a HuggingFace transformer model.\n\n        Args:\n            texts: The list of texts to embed.\n\n        Returns:\n            List of embeddings, one for each text.\n        '
        texts = list(map(lambda x: x.replace('\n', ' '), texts))
        embeddings = [self.embed(text, **self.encode_kwargs).tolist() for text in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        if False:
            return 10
        'Compute query embeddings using a bigdl-llm transformer model.\n\n        Args:\n            text: The text to embed.\n\n        Returns:\n            Embeddings for the text.\n        '
        text = text.replace('\n', ' ')
        embedding = self.embed(text, **self.encode_kwargs)
        return embedding.tolist()