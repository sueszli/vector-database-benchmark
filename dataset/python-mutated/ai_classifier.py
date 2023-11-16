import inspect
from enum import Enum
from functools import partial, wraps
from types import GenericAlias
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Generic, Literal, Optional, TypeVar, Union, cast, get_args, get_origin, overload
from pydantic import BaseModel, Field, TypeAdapter
from typing_extensions import ParamSpec, Self
from marvin.components.prompt import PromptFunction
from marvin.serializers import create_vocabulary_from_type
from marvin.settings import settings
from marvin.utilities.jinja import BaseEnvironment
if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion
T = TypeVar('T', bound=Union[GenericAlias, type, list[str]])
P = ParamSpec('P')

class AIClassifier(BaseModel, Generic[P, T]):
    fn: Optional[Callable[P, T]] = None
    environment: Optional[BaseEnvironment] = None
    prompt: Optional[str] = Field(default=inspect.cleandoc('You are an expert classifier that always choose correctly. \n- {{_doc}} \n- You must classify `{{text}}` into one of the following classes:{% for option in _options %}    Class {{ loop.index - 1}} (value: {{ option }}){% endfor %}ASSISTANT: The correct class label is Class'))
    enumerate: bool = True
    encoder: Callable[[str], list[int]] = Field(default=None)
    max_tokens: Optional[int] = 1
    render_kwargs: dict[str, Any] = Field(default_factory=dict)
    create: Optional[Callable[..., 'ChatCompletion']] = Field(default=None)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> list[T]:
        if False:
            for i in range(10):
                print('nop')
        create = self.create
        if self.fn is None:
            raise NotImplementedError
        if create is None:
            from marvin.settings import settings
            create = settings.openai.chat.completions.create
        return self.parse(create(**self.as_prompt(*args, **kwargs).serialize()))

    def parse(self, response: 'ChatCompletion') -> list[T]:
        if False:
            while True:
                i = 10
        if not response.choices[0].message.content:
            raise ValueError(f'Expected a response, got {response.choices[0].message.content}')
        _response: list[int] = [int(index) for index in list(response.choices[0].message.content)]
        _return: T = cast(T, self.fn.__annotations__.get('return'))
        _vocabulary: list[str] = create_vocabulary_from_type(_return)
        if isinstance(_return, list) and next(iter(get_args(list[str])), None) == str:
            return cast(list[T], [_vocabulary[int(index)] for index in _response])
        elif get_origin(_return) == Literal:
            return [TypeAdapter(_return).validate_python(_vocabulary[int(index)]) for index in _response]
        elif isinstance(_return, type) and issubclass(_return, Enum):
            return [TypeAdapter(_return).validate_python(1 + int(index)) for index in _response]
        raise TypeError(f'Expected Literal or Enum or list[str], got {type(_return)} with value {_return}')

    def as_prompt(self, *args: P.args, **kwargs: P.kwargs) -> PromptFunction[BaseModel]:
        if False:
            i = 10
            return i + 15
        return PromptFunction[BaseModel].as_grammar(fn=self.fn, environment=self.environment, prompt=self.prompt, enumerate=self.enumerate, encoder=self.encoder, max_tokens=self.max_tokens, **self.render_kwargs)(*args, **kwargs)

    @overload
    @classmethod
    def as_decorator(cls: type[Self], *, environment: Optional[BaseEnvironment]=None, prompt: Optional[str]=None, enumerate: bool=True, encoder: Callable[[str], list[int]]=settings.openai.chat.completions.encoder, max_tokens: Optional[int]=1, acreate: Optional[Callable[..., Awaitable[Any]]]=None, **render_kwargs: Any) -> Callable[P, Self]:
        if False:
            i = 10
            return i + 15
        pass

    @overload
    @classmethod
    def as_decorator(cls: type[Self], fn: Callable[P, T], *, environment: Optional[BaseEnvironment]=None, prompt: Optional[str]=None, enumerate: bool=True, encoder: Callable[[str], list[int]]=settings.openai.chat.completions.encoder, max_tokens: Optional[int]=1, acreate: Optional[Callable[..., Awaitable[Any]]]=None, **render_kwargs: Any) -> Self:
        if False:
            print('Hello World!')
        pass

    @classmethod
    def as_decorator(cls: type[Self], fn: Optional[Callable[P, T]]=None, *, environment: Optional[BaseEnvironment]=None, prompt: Optional[str]=None, enumerate: bool=True, encoder: Callable[[str], list[int]]=settings.openai.chat.completions.encoder, max_tokens: Optional[int]=1, acreate: Optional[Callable[..., Awaitable[Any]]]=None, **render_kwargs: Any) -> Union[Self, Callable[[Callable[P, T]], Self]]:
        if False:
            for i in range(10):
                print('nop')
        if fn is None:
            return partial(cls, environment=environment, prompt=prompt, enumerate=enumerate, encoder=encoder, max_tokens=max_tokens, acreate=acreate, **{'prompt': prompt} if prompt else {}, **render_kwargs)
        return cls(fn=fn, environment=environment, enumerate=enumerate, encoder=encoder, max_tokens=max_tokens, **{'prompt': prompt} if prompt else {}, **render_kwargs)

@overload
def ai_classifier(*, environment: Optional[BaseEnvironment]=None, prompt: Optional[str]=None, enumerate: bool=True, encoder: Callable[[str], list[int]]=settings.openai.chat.completions.encoder, max_tokens: Optional[int]=1, **render_kwargs: Any) -> Callable[[Callable[P, T]], Callable[P, T]]:
    if False:
        i = 10
        return i + 15
    pass

@overload
def ai_classifier(fn: Callable[P, T], *, environment: Optional[BaseEnvironment]=None, prompt: Optional[str]=None, enumerate: bool=True, encoder: Callable[[str], list[int]]=settings.openai.chat.completions.encoder, max_tokens: Optional[int]=1, **render_kwargs: Any) -> Callable[P, T]:
    if False:
        return 10
    pass

def ai_classifier(fn: Optional[Callable[P, T]]=None, *, environment: Optional[BaseEnvironment]=None, prompt: Optional[str]=None, enumerate: bool=True, encoder: Callable[[str], list[int]]=settings.openai.chat.completions.encoder, max_tokens: Optional[int]=1, **render_kwargs: Any) -> Union[Callable[[Callable[P, T]], Callable[P, T]], Callable[P, T]]:
    if False:
        while True:
            i = 10

    def wrapper(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        if False:
            i = 10
            return i + 15
        return AIClassifier[P, T].as_decorator(func, environment=environment, prompt=prompt, enumerate=enumerate, encoder=encoder, max_tokens=max_tokens, **render_kwargs)(*args, **kwargs)[0]
    if fn is not None:
        return wraps(fn)(partial(wrapper, fn))

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        if False:
            i = 10
            return i + 15
        return wraps(fn)(partial(wrapper, fn))
    return decorator