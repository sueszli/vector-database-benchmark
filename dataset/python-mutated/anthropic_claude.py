import os
from typing import Dict, List, Union, Optional
import json
import logging
import requests
import requests_cache
import sseclient
from tokenizers import Tokenizer, Encoding
from haystack.errors import AnthropicError, AnthropicRateLimitError, AnthropicUnauthorizedError
from haystack.nodes.prompt.invocation_layer.base import PromptModelInvocationLayer
from haystack.nodes.prompt.invocation_layer.handlers import TokenStreamingHandler, DefaultTokenStreamingHandler
from haystack.utils import request_with_retry
from haystack.environment import HAYSTACK_REMOTE_API_MAX_RETRIES, HAYSTACK_REMOTE_API_TIMEOUT_SEC
ANTHROPIC_TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))
ANTHROPIC_MAX_RETRIES = int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5))
logger = logging.getLogger(__name__)
CLAUDE_TOKENIZER_REMOTE_FILE = 'https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/src/anthropic/tokenizer.json'

class AnthropicClaudeInvocationLayer(PromptModelInvocationLayer):
    """
    Anthropic Claude Invocation Layer
    This layer invokes the Claude API provided by Anthropic.
    """

    def __init__(self, api_key: str, model_name_or_path: str='claude-2', max_length=200, **kwargs):
        if False:
            while True:
                i = 10
        "\n         Creates an instance of PromptModelInvocation Layer for Claude models by Anthropic.\n        :param model_name_or_path: The name or path of the underlying model.\n        :param max_tokens_to_sample: The maximum length of the output text.\n        :param api_key: The Anthropic API key.\n        :param kwargs: Additional keyword arguments passed to the underlying model. The list of Anthropic-relevant\n        kwargs includes: stop_sequences, temperature, top_p, top_k, and stream. For more details about these kwargs,\n        see Anthropic's [documentation](https://docs.anthropic.com/claude/reference/complete_post).\n        "
        super().__init__(model_name_or_path)
        if not isinstance(api_key, str) or len(api_key) == 0:
            raise AnthropicError(f'api_key {api_key} must be a valid Anthropic key. Visit https://console.anthropic.com/account/keys to get one.')
        self.api_key = api_key
        self.max_length = max_length
        supported_kwargs = ['temperature', 'top_p', 'top_k', 'stop_sequences', 'stream', 'stream_handler']
        self.model_input_kwargs = {k: v for (k, v) in kwargs.items() if k in supported_kwargs}
        self.max_tokens_limit = kwargs.get('model_max_length', 100000)
        self.tokenizer: Tokenizer = self._init_tokenizer()

    def _init_tokenizer(self) -> Tokenizer:
        if False:
            print('Hello World!')
        expire_after = 60 * 60 * 60 * 24
        with requests_cache.enabled(expire_after=expire_after):
            res = request_with_retry(method='GET', url=CLAUDE_TOKENIZER_REMOTE_FILE)
            res.raise_for_status()
        return Tokenizer.from_str(res.text)

    def invoke(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Invokes a prompt on the model. It takes in a prompt and returns a list of responses using a REST invocation.\n        :return: The responses are being returned.\n        '
        human_prompt = '\n\nHuman: '
        assistant_prompt = '\n\nAssistant: '
        prompt = kwargs.get('prompt')
        if not prompt:
            raise ValueError(f'No prompt provided. Model {self.model_name_or_path} requires prompt.Make sure to provide prompt in kwargs.')
        kwargs_with_defaults = self.model_input_kwargs
        if 'stop_sequence' in kwargs:
            kwargs['stop_words'] = kwargs.pop('stop_sequence')
        if 'max_tokens_to_sample' in kwargs:
            kwargs['max_length'] = kwargs.pop('max_tokens_to_sample')
        kwargs_with_defaults.update(kwargs)
        stream = kwargs_with_defaults.get('stream', False) or kwargs_with_defaults.get('stream_handler', None) is not None
        stop_words = kwargs_with_defaults.get('stop_words') or [human_prompt]
        if human_prompt not in stop_words:
            stop_words.append(human_prompt)
        prompt = f'{human_prompt}{prompt}{assistant_prompt}'
        data = {'model': self.model_name_or_path, 'prompt': prompt, 'max_tokens_to_sample': kwargs_with_defaults.get('max_length', self.max_length), 'temperature': kwargs_with_defaults.get('temperature', 1), 'top_p': kwargs_with_defaults.get('top_p', -1), 'top_k': kwargs_with_defaults.get('top_k', -1), 'stream': stream, 'stop_sequences': stop_words}
        if not stream:
            res = self._post(data=data)
            return [res.json()['completion'].strip()]
        res = self._post(data=data, stream=True)
        handler: TokenStreamingHandler = kwargs_with_defaults.pop('stream_handler', DefaultTokenStreamingHandler())
        client = sseclient.SSEClient(res)
        tokens = []
        try:
            for event in client.events():
                ed = json.loads(event.data)
                if 'completion' in ed:
                    tokens.append(handler(ed['completion']))
        finally:
            client.close()
        return [''.join(tokens)]

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        if False:
            i = 10
            return i + 15
        'Make sure the length of the prompt and answer is within the max tokens limit of the model.\n        If needed, truncate the prompt text so that it fits within the limit.\n        :param prompt: Prompt text to be sent to the generative model.\n        '
        if isinstance(prompt, List):
            raise ValueError("Anthropic invocation layer doesn't support a dictionary as prompt")
        token_limit = self.max_tokens_limit - self.max_length
        self.tokenizer.enable_truncation(token_limit)
        encoded_prompt: Encoding = self.tokenizer.encode(prompt.split(' '), is_pretokenized=True)
        if encoded_prompt.overflowing:
            logger.warning('The prompt has been truncated from %s tokens to %s tokens so that the prompt length and answer length (%s tokens) fits within the max token limit (%s tokens). Reduce the length of the prompt to prevent it from being cut off.', len(encoded_prompt.ids) + len(encoded_prompt.overflowing), self.max_tokens_limit - self.max_length, self.max_length, self.max_tokens_limit)
        return ' '.join(encoded_prompt.tokens)

    def _post(self, data: Dict, attempts: int=ANTHROPIC_MAX_RETRIES, status_codes_to_retry: Optional[List[int]]=None, timeout: float=ANTHROPIC_TIMEOUT, **kwargs):
        if False:
            print('Hello World!')
        '\n        Post data to Anthropic.\n        Retries request in case it fails with any code in status_codes_to_retry\n        or with timeout.\n        All kwargs are passed to ``requests.request``, so it accepts the same arguments.\n        Returns a ``requests.Response`` object.\n\n        :param data: Object to send in the body of the request.\n        :param attempts: Number of times to attempt a request in case of failures, defaults to 5.\n        :param timeout: Number of seconds to wait for the server to send data before giving up, defaults to 30.\n        :raises AnthropicRateLimitError: Raised if a request fails with the 429 status code.\n        :raises AnthropicUnauthorizedError: Raised if a request fails with the 401 status code.\n        :raises AnthropicError: Raised if requests fail for any other reason.\n        :return: :class:`Response <Response>` object\n        '
        if status_codes_to_retry is None:
            status_codes_to_retry = [429]
        try:
            response = request_with_retry(attempts=attempts, status_codes_to_retry=status_codes_to_retry, method='POST', url='https://api.anthropic.com/v1/complete', headers={'x-api-key': self.api_key, 'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'}, data=json.dumps(data), timeout=timeout, **kwargs)
        except requests.HTTPError as err:
            res = err.response
            if res.status_code == 429:
                raise AnthropicRateLimitError(f'API rate limit exceeded: {res.text}')
            if res.status_code == 401:
                raise AnthropicUnauthorizedError(f'API key is invalid: {res.text}')
            raise AnthropicError(f'Anthropic returned an error.\nStatus code: {res.status_code}\nResponse body: {res.text}', status_code=res.status_code)
        return response

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        if False:
            while True:
                i = 10
        '\n        Ensures Anthropic Claude Invocation Layer is selected only when Claude models are specified in\n        the model name.\n        '
        return model_name_or_path.startswith('claude-')