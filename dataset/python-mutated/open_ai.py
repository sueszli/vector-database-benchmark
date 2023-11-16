from typing import List, Union, Dict, Optional, cast, Any
import json
import logging
import sseclient
from haystack.errors import OpenAIError
from haystack.nodes.prompt.invocation_layer.utils import has_azure_parameters
from haystack.utils.openai_utils import _openai_text_completion_tokenization_details, load_openai_tokenizer, _check_openai_finish_reason, check_openai_async_policy_violation, check_openai_policy_violation, openai_async_request, openai_request
from haystack.nodes.prompt.invocation_layer.base import PromptModelInvocationLayer
from haystack.nodes.prompt.invocation_layer.handlers import TokenStreamingHandler, DefaultTokenStreamingHandler
logger = logging.getLogger(__name__)

class OpenAIInvocationLayer(PromptModelInvocationLayer):
    """
    PromptModelInvocationLayer implementation for OpenAI's GPT-3 InstructGPT models. Invocations are made using REST API.
    See [OpenAI GPT-3](https://platform.openai.com/docs/models/gpt-3) for more details.

    Note: kwargs other than init parameter names are ignored to enable reflective construction of the class
    as many variants of PromptModelInvocationLayer are possible and they may have different parameters.
    """

    def __init__(self, api_key: str, model_name_or_path: str='text-davinci-003', max_length: Optional[int]=100, api_base: str='https://api.openai.com/v1', openai_organization: Optional[str]=None, **kwargs):
        if False:
            return 10
        "\n         Creates an instance of OpenAIInvocationLayer for OpenAI's GPT-3 InstructGPT models.\n\n        :param model_name_or_path: The name or path of the underlying model.\n        :param max_length: The maximum number of tokens the output text can have.\n        :param api_key: The OpenAI API key.\n        :param api_base: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.\n        :param openai_organization: The OpenAI-Organization ID, defaults to `None`. For more details, see see OpenAI\n        [documentation](https://platform.openai.com/docs/api-reference/requesting-organization).\n        :param kwargs: Additional keyword arguments passed to the underlying model. Due to reflective construction of\n        all PromptModelInvocationLayer instances, this instance of OpenAIInvocationLayer might receive some unrelated\n        kwargs. Only the kwargs relevant to OpenAIInvocationLayer are considered. The list of OpenAI-relevant\n        kwargs includes: suffix, temperature, top_p, presence_penalty, frequency_penalty, best_of, n, max_tokens,\n        logit_bias, stop, echo, and logprobs. For more details about these kwargs, see OpenAI\n        [documentation](https://platform.openai.com/docs/api-reference/completions/create).\n        Note: additional model argument moderate_content will filter input and generated answers for potentially\n        sensitive content using the [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation)\n        if set. If the input or answers are flagged, an empty list is returned in place of the answers.\n        "
        super().__init__(model_name_or_path)
        if not isinstance(api_key, str) or len(api_key) == 0:
            raise OpenAIError(f'api_key {api_key} must be a valid OpenAI key. Visit https://openai.com/api/ to get one.')
        self.api_key = api_key
        self.api_base = api_base
        self.openai_organization = openai_organization
        self.max_length = max_length or 16
        self.model_input_kwargs = {key: kwargs[key] for key in ['suffix', 'max_tokens', 'temperature', 'top_p', 'n', 'logprobs', 'echo', 'stop', 'presence_penalty', 'frequency_penalty', 'best_of', 'logit_bias', 'stream', 'stream_handler', 'moderate_content'] if key in kwargs}
        (tokenizer_name, max_tokens_limit) = _openai_text_completion_tokenization_details(model_name=self.model_name_or_path)
        self.max_tokens_limit = max_tokens_limit
        self._tokenizer = load_openai_tokenizer(tokenizer_name=tokenizer_name)

    @property
    def url(self) -> str:
        if False:
            while True:
                i = 10
        return f'{self.api_base}/completions'

    @property
    def headers(self) -> Dict[str, str]:
        if False:
            while True:
                i = 10
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        if self.openai_organization:
            headers['OpenAI-Organization'] = self.openai_organization
        return headers

    def _prepare_invoke(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        prompt = kwargs.get('prompt')
        if not prompt:
            raise ValueError(f'No prompt provided. Model {self.model_name_or_path} requires prompt.Make sure to provide prompt in kwargs.')
        kwargs_with_defaults = self.model_input_kwargs
        if kwargs:
            if 'stop_words' in kwargs:
                kwargs['stop'] = kwargs.pop('stop_words')
            if 'top_k' in kwargs:
                top_k = kwargs.pop('top_k')
                kwargs['n'] = top_k
                kwargs['best_of'] = top_k
            kwargs_with_defaults.update(kwargs)
        stream = kwargs_with_defaults.get('stream', False) or kwargs_with_defaults.get('stream_handler', None) is not None
        moderation = kwargs_with_defaults.get('moderate_content', False)
        base_payload = {'model': self.model_name_or_path, 'max_tokens': kwargs_with_defaults.get('max_tokens', self.max_length), 'temperature': kwargs_with_defaults.get('temperature', 0.7), 'top_p': kwargs_with_defaults.get('top_p', 1), 'n': kwargs_with_defaults.get('n', 1), 'stream': stream, 'stop': kwargs_with_defaults.get('stop', None), 'presence_penalty': kwargs_with_defaults.get('presence_penalty', 0), 'frequency_penalty': kwargs_with_defaults.get('frequency_penalty', 0), 'logit_bias': kwargs_with_defaults.get('logit_bias', {})}
        return (prompt, base_payload, kwargs_with_defaults, stream, moderation)

    def invoke(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Invokes a prompt on the model. Based on the model, it takes in a prompt (or either a prompt or a list of messages)\n        and returns a list of responses using a REST invocation.\n\n        :return: The responses are being returned.\n\n        Note: Only kwargs relevant to OpenAI are passed to OpenAI rest API. Others kwargs are ignored.\n        For more details, see OpenAI [documentation](https://platform.openai.com/docs/api-reference/completions/create).\n        '
        (prompt, base_payload, kwargs_with_defaults, stream, moderation) = self._prepare_invoke(*args, **kwargs)
        if moderation and check_openai_policy_violation(input=prompt, headers=self.headers):
            logger.info("Prompt '%s' will not be sent to OpenAI due to potential policy violation.", prompt)
            return []
        extra_payload = {'prompt': prompt, 'suffix': kwargs_with_defaults.get('suffix', None), 'logprobs': kwargs_with_defaults.get('logprobs', None), 'echo': kwargs_with_defaults.get('echo', False), 'best_of': kwargs_with_defaults.get('best_of', 1)}
        payload = {**base_payload, **extra_payload}
        if not stream:
            res = openai_request(url=self.url, headers=self.headers, payload=payload)
            _check_openai_finish_reason(result=res, payload=payload)
            responses = [ans['text'].strip() for ans in res['choices']]
        else:
            response = openai_request(url=self.url, headers=self.headers, payload=payload, read_response=False, stream=True)
            handler: TokenStreamingHandler = kwargs_with_defaults.pop('stream_handler', DefaultTokenStreamingHandler())
            responses = self._process_streaming_response(response=response, stream_handler=handler)
        if moderation and check_openai_policy_violation(input=responses, headers=self.headers):
            logger.info("Response '%s' will not be returned due to potential policy violation.", responses)
            return []
        return responses

    async def ainvoke(self, *args, **kwargs):
        """
        asyncio version of the `invoke` method.
        """
        (prompt, base_payload, kwargs_with_defaults, stream, moderation) = self._prepare_invoke(*args, **kwargs)
        if moderation and await check_openai_async_policy_violation(input=prompt, headers=self.headers):
            logger.info("Prompt '%s' will not be sent to OpenAI due to potential policy violation.", prompt)
            return []
        extra_payload = {'prompt': prompt, 'suffix': kwargs_with_defaults.get('suffix', None), 'logprobs': kwargs_with_defaults.get('logprobs', None), 'echo': kwargs_with_defaults.get('echo', False), 'best_of': kwargs_with_defaults.get('best_of', 1)}
        payload = {**base_payload, **extra_payload}
        if not stream:
            res = await openai_async_request(url=self.url, headers=self.headers, payload=payload)
            _check_openai_finish_reason(result=res, payload=payload)
            responses = [ans['text'].strip() for ans in res['choices']]
        else:
            response = await openai_async_request(url=self.url, headers=self.headers, payload=payload, read_response=False, stream=True)
            handler: TokenStreamingHandler = kwargs_with_defaults.pop('stream_handler', DefaultTokenStreamingHandler())
            responses = self._process_streaming_response(response=response, stream_handler=handler)
        if moderation and await check_openai_async_policy_violation(input=responses, headers=self.headers):
            logger.info("Response '%s' will not be returned due to potential policy violation.", responses)
            return []
        return responses

    def _process_streaming_response(self, response, stream_handler: TokenStreamingHandler):
        if False:
            print('Hello World!')
        client = sseclient.SSEClient(response)
        tokens: List[str] = []
        try:
            for event in client.events():
                if event.data != TokenStreamingHandler.DONE_MARKER:
                    event_data = json.loads(event.data)
                    token: str = self._extract_token(event_data)
                    if token:
                        tokens.append(stream_handler(token, event_data=event_data['choices']))
        finally:
            client.close()
        return [''.join(tokens)]

    def _extract_token(self, event_data: Dict[str, Any]):
        if False:
            for i in range(10):
                print('nop')
        return event_data['choices'][0]['text']

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        if False:
            for i in range(10):
                print('nop')
        'Ensure that the length of the prompt and answer is within the max tokens limit of the model.\n        If needed, truncate the prompt text so that it fits within the limit.\n\n        :param prompt: Prompt text to be sent to the generative model.\n        '
        n_prompt_tokens = len(self._tokenizer.encode(cast(str, prompt)))
        n_answer_tokens = self.max_length
        if n_prompt_tokens + n_answer_tokens <= self.max_tokens_limit:
            return prompt
        logger.warning('The prompt has been truncated from %s tokens to %s tokens so that the prompt length and answer length (%s tokens) fit within the max token limit (%s tokens). Reduce the length of the prompt to prevent it from being cut off.', n_prompt_tokens, self.max_tokens_limit - n_answer_tokens, n_answer_tokens, self.max_tokens_limit)
        tokenized_payload = self._tokenizer.encode(prompt)
        decoded_string = self._tokenizer.decode(tokenized_payload[:self.max_tokens_limit - n_answer_tokens])
        return decoded_string

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        if False:
            print('Hello World!')
        valid_model = model_name_or_path in ['ada', 'babbage', 'davinci', 'curie', 'gpt-3.5-turbo-instruct'] or any((m in model_name_or_path for m in ['-ada-', '-babbage-', '-davinci-', '-curie-']))
        return valid_model and (not has_azure_parameters(**kwargs))