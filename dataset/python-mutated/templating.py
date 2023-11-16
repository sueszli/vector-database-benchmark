import typing
import warnings
from os import PathLike
from starlette.background import BackgroundTask
from starlette.datastructures import URL
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.types import Receive, Scope, Send
try:
    import jinja2
    if hasattr(jinja2, 'pass_context'):
        pass_context = jinja2.pass_context
    else:
        pass_context = jinja2.contextfunction
except ModuleNotFoundError:
    jinja2 = None

class _TemplateResponse(HTMLResponse):

    def __init__(self, template: typing.Any, context: typing.Dict[str, typing.Any], status_code: int=200, headers: typing.Optional[typing.Mapping[str, str]]=None, media_type: typing.Optional[str]=None, background: typing.Optional[BackgroundTask]=None):
        if False:
            print('Hello World!')
        self.template = template
        self.context = context
        content = template.render(context)
        super().__init__(content, status_code, headers, media_type, background)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        request = self.context.get('request', {})
        extensions = request.get('extensions', {})
        if 'http.response.debug' in extensions:
            await send({'type': 'http.response.debug', 'info': {'template': self.template, 'context': self.context}})
        await super().__call__(scope, receive, send)

class Jinja2Templates:
    """
    templates = Jinja2Templates("templates")

    return templates.TemplateResponse("index.html", {"request": request})
    """

    @typing.overload
    def __init__(self, directory: 'typing.Union[str, PathLike[typing.AnyStr], typing.Sequence[typing.Union[str, PathLike[typing.AnyStr]]]]', *, context_processors: typing.Optional[typing.List[typing.Callable[[Request], typing.Dict[str, typing.Any]]]]=None, **env_options: typing.Any) -> None:
        if False:
            while True:
                i = 10
        ...

    @typing.overload
    def __init__(self, *, env: 'jinja2.Environment', context_processors: typing.Optional[typing.List[typing.Callable[[Request], typing.Dict[str, typing.Any]]]]=None) -> None:
        if False:
            print('Hello World!')
        ...

    def __init__(self, directory: 'typing.Union[str, PathLike[typing.AnyStr], typing.Sequence[typing.Union[str, PathLike[typing.AnyStr]]], None]'=None, *, context_processors: typing.Optional[typing.List[typing.Callable[[Request], typing.Dict[str, typing.Any]]]]=None, env: typing.Optional['jinja2.Environment']=None, **env_options: typing.Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        if env_options:
            warnings.warn('Extra environment options are deprecated. Use a preconfigured jinja2.Environment instead.', DeprecationWarning)
        assert jinja2 is not None, 'jinja2 must be installed to use Jinja2Templates'
        assert directory or env, "either 'directory' or 'env' arguments must be passed"
        self.context_processors = context_processors or []
        if directory is not None:
            self.env = self._create_env(directory, **env_options)
        elif env is not None:
            self.env = env
        self._setup_env_defaults(self.env)

    def _create_env(self, directory: 'typing.Union[str, PathLike[typing.AnyStr], typing.Sequence[typing.Union[str, PathLike[typing.AnyStr]]]]', **env_options: typing.Any) -> 'jinja2.Environment':
        if False:
            i = 10
            return i + 15
        loader = jinja2.FileSystemLoader(directory)
        env_options.setdefault('loader', loader)
        env_options.setdefault('autoescape', True)
        return jinja2.Environment(**env_options)

    def _setup_env_defaults(self, env: 'jinja2.Environment') -> None:
        if False:
            return 10

        @pass_context
        def url_for(context: typing.Dict[str, typing.Any], name: str, /, **path_params: typing.Any) -> URL:
            if False:
                while True:
                    i = 10
            request: Request = context['request']
            return request.url_for(name, **path_params)
        env.globals.setdefault('url_for', url_for)

    def get_template(self, name: str) -> 'jinja2.Template':
        if False:
            return 10
        return self.env.get_template(name)

    @typing.overload
    def TemplateResponse(self, request: Request, name: str, context: typing.Optional[typing.Dict[str, typing.Any]]=None, status_code: int=200, headers: typing.Optional[typing.Mapping[str, str]]=None, media_type: typing.Optional[str]=None, background: typing.Optional[BackgroundTask]=None) -> _TemplateResponse:
        if False:
            print('Hello World!')
        ...

    @typing.overload
    def TemplateResponse(self, name: str, context: typing.Optional[typing.Dict[str, typing.Any]]=None, status_code: int=200, headers: typing.Optional[typing.Mapping[str, str]]=None, media_type: typing.Optional[str]=None, background: typing.Optional[BackgroundTask]=None) -> _TemplateResponse:
        if False:
            for i in range(10):
                print('nop')
        ...

    def TemplateResponse(self, *args: typing.Any, **kwargs: typing.Any) -> _TemplateResponse:
        if False:
            return 10
        if args:
            if isinstance(args[0], str):
                warnings.warn('The `name` is not the first parameter anymore. The first parameter should be the `Request` instance.\nReplace `TemplateResponse(name, {"request": request})` by `TemplateResponse(request, name)`.', DeprecationWarning)
                name = args[0]
                context = args[1] if len(args) > 1 else kwargs.get('context', {})
                status_code = args[2] if len(args) > 2 else kwargs.get('status_code', 200)
                headers = args[2] if len(args) > 2 else kwargs.get('headers')
                media_type = args[3] if len(args) > 3 else kwargs.get('media_type')
                background = args[4] if len(args) > 4 else kwargs.get('background')
                if 'request' not in context:
                    raise ValueError('context must include a "request" key')
                request = context['request']
            else:
                request = args[0]
                name = args[1] if len(args) > 1 else kwargs['name']
                context = args[2] if len(args) > 2 else kwargs.get('context', {})
                status_code = args[3] if len(args) > 3 else kwargs.get('status_code', 200)
                headers = args[4] if len(args) > 4 else kwargs.get('headers')
                media_type = args[5] if len(args) > 5 else kwargs.get('media_type')
                background = args[6] if len(args) > 6 else kwargs.get('background')
        else:
            if 'request' not in kwargs:
                warnings.warn('The `TemplateResponse` now requires the `request` argument.\nReplace `TemplateResponse(name, {"context": context})` by `TemplateResponse(request, name)`.', DeprecationWarning)
                if 'request' not in kwargs.get('context', {}):
                    raise ValueError('context must include a "request" key')
            context = kwargs.get('context', {})
            request = kwargs.get('request', context.get('request'))
            name = typing.cast(str, kwargs['name'])
            status_code = kwargs.get('status_code', 200)
            headers = kwargs.get('headers')
            media_type = kwargs.get('media_type')
            background = kwargs.get('background')
        context.setdefault('request', request)
        for context_processor in self.context_processors:
            context.update(context_processor(request))
        template = self.get_template(name)
        return _TemplateResponse(template, context, status_code=status_code, headers=headers, media_type=media_type, background=background)