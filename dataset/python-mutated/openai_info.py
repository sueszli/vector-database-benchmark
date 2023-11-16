from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional, Generator
MODEL_COST_PER_1K_TOKENS = {'gpt-4': 0.03, 'gpt-4-0613': 0.03, 'gpt-4-32k': 0.06, 'gpt-4-32k-0613': 0.06, 'gpt-4-1106-preview': 0.01, 'gpt-4-completion': 0.06, 'gpt-4-0613-completion': 0.06, 'gpt-4-32k-completion': 0.12, 'gpt-4-32k-0613-completion': 0.12, 'gpt-4-1106-preview-completion': 0.03, 'gpt-3.5-turbo': 0.001, 'gpt-3.5-turbo-0613': 0.001, 'gpt-3.5-turbo-1106': 0.001, 'gpt-3.5-turbo-instruct': 0.0015, 'gpt-3.5-turbo-16k': 0.001, 'gpt-3.5-turbo-16k-0613': 0.001, 'gpt-3.5-turbo-completion': 0.002, 'gpt-3.5-turbo-0613-completion': 0.002, 'gpt-3.5-turbo-1106-completion': 0.002, 'gpt-3.5-turbo-instruct-completion': 0.003, 'gpt-3.5-turbo-16k-completion': 0.002, 'gpt-3.5-turbo-16k-0613-completion': 0.002, 'gpt-35-turbo': 0.0015, 'gpt-35-turbo-0613': 0.0015, 'gpt-35-turbo-instruct': 0.0015, 'gpt-35-turbo-16k': 0.003, 'gpt-35-turbo-16k-0613': 0.003, 'gpt-35-turbo-completion': 0.002, 'gpt-35-turbo-0613-completion': 0.002, 'gpt-35-turbo-instruct-completion': 0.002, 'gpt-35-turbo-16k-completion': 0.004, 'gpt-35-turbo-16k-0613-completion': 0.004, 'gpt-3.5-turbo-0613-finetuned': 0.012, 'gpt-3.5-turbo-0613-finetuned-completion': 0.016, 'gpt-35-turbo-0613-azure-finetuned': 0.0015, 'gpt-35-turbo-0613-azure-finetuned-completion': 0.002, 'text-davinci-003': 0.02}

def standardize_model_name(model_name: str, is_completion: bool=False) -> str:
    if False:
        return 10
    '\n    Standardize the model name to a format that can be used in the OpenAI API.\n\n    Args:\n        model_name: Model name to standardize.\n        is_completion: Whether the model is used for completion or not.\n            Defaults to False.\n\n    Returns:\n        Standardized model name.\n\n    '
    model_name = model_name.lower()
    if '.ft-' in model_name:
        model_name = model_name.split('.ft-')[0] + '-azure-finetuned'
    if 'ft:' in model_name:
        model_name = model_name.split(':')[1] + '-finetuned'
    if is_completion and (model_name.startswith('gpt-4') or model_name.startswith('gpt-3.5') or model_name.startswith('gpt-35') or ('finetuned' in model_name)):
        return f'{model_name}-completion'
    else:
        return model_name

def get_openai_token_cost_for_model(model_name: str, num_tokens: int, is_completion: bool=False) -> float:
    if False:
        i = 10
        return i + 15
    '\n    Get the cost in USD for a given model and number of tokens.\n\n    Args:\n        model_name (str): Name of the model\n        num_tokens (int): Number of tokens.\n        is_completion: Whether `num_tokens` refers to completion tokens or not.\n            Defaults to False.\n\n    Returns:\n        float: Cost in USD.\n    '
    model_name = standardize_model_name(model_name, is_completion=is_completion)
    if model_name not in MODEL_COST_PER_1K_TOKENS:
        raise ValueError(f'Unknown model: {model_name}. Please provide a valid OpenAI model name.Known models are: ' + ', '.join(MODEL_COST_PER_1K_TOKENS.keys()))
    return MODEL_COST_PER_1K_TOKENS[model_name] * (num_tokens / 1000)

class OpenAICallbackHandler:
    """Callback Handler that tracks OpenAI info."""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'Tokens Used: {self.total_tokens}\n\tPrompt Tokens: {self.prompt_tokens}\n\tCompletion Tokens: {self.completion_tokens}\nTotal Cost (USD): ${self.total_cost:9.6f}'

    def __call__(self, response) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Collect token usage'
        usage = response.usage
        if not hasattr(usage, 'total_tokens'):
            return None
        model_name = standardize_model_name(response.model)
        if model_name in MODEL_COST_PER_1K_TOKENS:
            prompt_cost = get_openai_token_cost_for_model(model_name, usage.prompt_tokens)
            completion_cost = get_openai_token_cost_for_model(model_name, usage.completion_tokens, is_completion=True)
            self.total_cost += prompt_cost + completion_cost
        self.total_tokens += usage.total_tokens
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens

    def __copy__(self) -> 'OpenAICallbackHandler':
        if False:
            while True:
                i = 10
        'Return a copy of the callback handler.'
        return self
openai_callback_var: ContextVar[Optional[OpenAICallbackHandler]] = ContextVar('openai_callback', default=None)

@contextmanager
def get_openai_callback() -> Generator[OpenAICallbackHandler, None, None]:
    if False:
        return 10
    'Get the OpenAI callback handler in a context manager.\n    which conveniently exposes token and cost information.\n\n    Yields:\n        OpenAICallbackHandler: The OpenAI callback handler.\n\n    Example:\n        >>> with get_openai_callback() as cb:\n        ...     # Use the OpenAI callback handler\n    '
    cb = OpenAICallbackHandler()
    openai_callback_var.set(cb)
    yield cb
    openai_callback_var.set(None)