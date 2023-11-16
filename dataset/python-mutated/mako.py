from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING, Any, Mapping, TypeVar
from typing_extensions import ParamSpec
from litestar.exceptions import ImproperlyConfiguredException, MissingDependencyException, TemplateNotFoundException
from litestar.template.base import TemplateCallableType, TemplateEngineProtocol, TemplateProtocol, csrf_token, url_for, url_for_static_asset
try:
    from mako.exceptions import TemplateLookupException as MakoTemplateNotFound
    from mako.lookup import TemplateLookup
except ImportError as e:
    raise MissingDependencyException('mako') from e
if TYPE_CHECKING:
    from pathlib import Path
    from mako.template import Template as _MakoTemplate
__all__ = ('MakoTemplate', 'MakoTemplateEngine')
P = ParamSpec('P')
T = TypeVar('T')

class MakoTemplate(TemplateProtocol):
    """Mako template, implementing ``TemplateProtocol``"""

    def __init__(self, template: _MakoTemplate, template_callables: list[tuple[str, TemplateCallableType]]) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize a template.\n\n        Args:\n            template: Base ``MakoTemplate`` used by the underlying mako-engine\n            template_callables: List of callables passed to the template\n        '
        super().__init__()
        self.template = template
        self.template_callables = template_callables

    def render(self, *args: Any, **kwargs: Any) -> str:
        if False:
            while True:
                i = 10
        'Render a template.\n\n        Args:\n            args: Positional arguments passed to the engines ``render`` function\n            kwargs: Keyword arguments passed to the engines ``render`` function\n\n        Returns:\n            Rendered template as a string\n        '
        for (callable_key, template_callable) in self.template_callables:
            kwargs_copy = {**kwargs}
            kwargs[callable_key] = partial(template_callable, kwargs_copy)
        return str(self.template.render(*args, **kwargs))

class MakoTemplateEngine(TemplateEngineProtocol[MakoTemplate, Mapping[str, Any]]):
    """Mako based TemplateEngine."""

    def __init__(self, directory: Path | list[Path] | None=None, engine_instance: Any | None=None) -> None:
        if False:
            print('Hello World!')
        'Initialize template engine.\n\n        Args:\n            directory: Direct path or list of directory paths from which to serve templates.\n            engine_instance: A mako TemplateLookup instance.\n        '
        super().__init__(directory, engine_instance)
        if directory and engine_instance:
            raise ImproperlyConfiguredException('You must provide either a directory or a mako TemplateLookup.')
        if directory:
            self.engine = TemplateLookup(directories=directory if isinstance(directory, (list, tuple)) else [directory], default_filters=['h'])
        elif engine_instance:
            self.engine = engine_instance
        self._template_callables: list[tuple[str, TemplateCallableType]] = []
        self.register_template_callable(key='url_for_static_asset', template_callable=url_for_static_asset)
        self.register_template_callable(key='csrf_token', template_callable=csrf_token)
        self.register_template_callable(key='url_for', template_callable=url_for)

    def get_template(self, template_name: str) -> MakoTemplate:
        if False:
            for i in range(10):
                print('nop')
        'Retrieve a template by matching its name (dotted path) with files in the directory or directories provided.\n\n        Args:\n            template_name: A dotted path\n\n        Returns:\n            MakoTemplate instance\n\n        Raises:\n            TemplateNotFoundException: if no template is found.\n        '
        try:
            return MakoTemplate(template=self.engine.get_template(template_name), template_callables=self._template_callables)
        except MakoTemplateNotFound as exc:
            raise TemplateNotFoundException(template_name=template_name) from exc

    def register_template_callable(self, key: str, template_callable: TemplateCallableType[Mapping[str, Any], P, T]) -> None:
        if False:
            i = 10
            return i + 15
        'Register a callable on the template engine.\n\n        Args:\n            key: The callable key, i.e. the value to use inside the template to call the callable.\n            template_callable: A callable to register.\n\n        Returns:\n            None\n        '
        self._template_callables.append((key, template_callable))

    @classmethod
    def from_template_lookup(cls, template_lookup: TemplateLookup) -> MakoTemplateEngine:
        if False:
            while True:
                i = 10
        'Create a template engine from an existing mako TemplateLookup instance.\n\n        Args:\n            template_lookup: A mako TemplateLookup instance.\n\n        Returns:\n            MakoTemplateEngine instance\n        '
        return cls(directory=None, engine_instance=template_lookup)