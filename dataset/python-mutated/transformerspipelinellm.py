import importlib.util
import logging
from typing import Any, List, Mapping, Optional
from pydantic import Extra
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
DEFAULT_MODEL_ID = 'gpt2'
DEFAULT_TASK = 'text-generation'
VALID_TASKS = ('text2text-generation', 'text-generation', 'summarization')

class TransformersPipelineLLM(LLM):
    """Wrapper around the BigDL-LLM Transformer-INT4 model in Transformer.pipeline()

    Example:
        .. code-block:: python

            from langchain.llms import TransformersPipelineLLM
            llm = TransformersPipelineLLM.from_model_id(model_id="decapoda-research/llama-7b-hf")
    """
    pipeline: Any
    model_id: str = DEFAULT_MODEL_ID
    'Model name or model path to use.'
    model_kwargs: Optional[dict] = None
    'Key word arguments passed to the model.'
    pipeline_kwargs: Optional[dict] = None
    'Key word arguments passed to the pipeline.'

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @classmethod
    def from_model_id(cls, model_id: str, task: str, model_kwargs: Optional[dict]=None, pipeline_kwargs: Optional[dict]=None, **kwargs: Any) -> LLM:
        if False:
            while True:
                i = 10
        'Construct the pipeline object from model_id and task.'
        try:
            from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
            from transformers import AutoTokenizer, LlamaTokenizer
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ValueError('Could not import transformers python package. Please install it with `pip install transformers`.')
        _model_kwargs = model_kwargs or {}
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
        except:
            tokenizer = LlamaTokenizer.from_pretrained(model_id, **_model_kwargs)
        try:
            if task == 'text-generation':
                model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, **_model_kwargs)
            elif task in ('text2text-generation', 'summarization'):
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_4bit=True, **_model_kwargs)
            else:
                raise ValueError(f'Got invalid task {task}, currently only {VALID_TASKS} are supported')
        except ImportError as e:
            raise ValueError(f'Could not load the {task} model due to missing dependencies.') from e
        if 'trust_remote_code' in _model_kwargs:
            _model_kwargs = {k: v for (k, v) in _model_kwargs.items() if k != 'trust_remote_code'}
        _pipeline_kwargs = pipeline_kwargs or {}
        pipeline = hf_pipeline(task=task, model=model, tokenizer=tokenizer, device='cpu', model_kwargs=_model_kwargs, **_pipeline_kwargs)
        if pipeline.task not in VALID_TASKS:
            raise ValueError(f'Got invalid task {pipeline.task}, currently only {VALID_TASKS} are supported')
        return cls(pipeline=pipeline, model_id=model_id, model_kwargs=_model_kwargs, pipeline_kwargs=_pipeline_kwargs, **kwargs)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        'Get the identifying parameters.'
        return {'model_id': self.model_id, 'model_kwargs': self.model_kwargs, 'pipeline_kwargs': self.pipeline_kwargs}

    @property
    def _llm_type(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'BigDL-llm'

    def _call(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> str:
        if False:
            print('Hello World!')
        response = self.pipeline(prompt)
        if self.pipeline.task == 'text-generation':
            text = response[0]['generated_text'][len(prompt):]
        elif self.pipeline.task == 'text2text-generation':
            text = response[0]['generated_text']
        elif self.pipeline.task == 'summarization':
            text = response[0]['summary_text']
        else:
            raise ValueError(f'Got invalid task {self.pipeline.task}, currently only {VALID_TASKS} are supported')
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text