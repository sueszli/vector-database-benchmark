from typing import Any, Dict, List, Optional
import json
import numpy as np
from pydantic import BaseModel, Extra, root_validator
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from huggingface_hub import InferenceClient
HOSTED_INFERENCE_API = 'hosted_inference_api'
INFERENCE_ENDPOINTS = 'inference_endpoints'

class HuggingfaceHubEmbeddings(BaseModel, Embeddings):
    client: Any
    model: str
    huggingface_namespace: Optional[str] = None
    task_type: Optional[str] = None
    huggingfacehub_api_type: Optional[str] = None
    huggingfacehub_api_token: Optional[str] = None
    huggingfacehub_endpoint_url: Optional[str] = None

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        if False:
            print('Hello World!')
        values['huggingfacehub_api_token'] = get_from_dict_or_env(values, 'huggingfacehub_api_token', 'HUGGINGFACEHUB_API_TOKEN')
        values['client'] = InferenceClient(token=values['huggingfacehub_api_token'])
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if False:
            while True:
                i = 10
        model = ''
        if self.huggingfacehub_api_type == HOSTED_INFERENCE_API:
            model = self.model
        else:
            model = self.huggingfacehub_endpoint_url
        output = self.client.post(json={'inputs': texts, 'options': {'wait_for_model': False, 'use_cache': False}}, model=model)
        embeddings = json.loads(output.decode())
        return self.mean_pooling(embeddings)

    def embed_query(self, text: str) -> List[float]:
        if False:
            print('Hello World!')
        return self.embed_documents([text])[0]

    def mean_pooling(self, embeddings: List) -> List[float]:
        if False:
            print('Hello World!')
        if not isinstance(embeddings[0][0], list):
            return embeddings
        sentence_embeddings = [np.mean(embedding[0], axis=0).tolist() for embedding in embeddings]
        return sentence_embeddings