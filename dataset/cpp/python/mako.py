from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Mapping, TypeVar

from typing_extensions import ParamSpec

from litestar.exceptions import ImproperlyConfiguredException, MissingDependencyException, TemplateNotFoundException
from litestar.template.base import (
    TemplateCallableType,
    TemplateEngineProtocol,
    TemplateProtocol,
    csrf_token,
    url_for,
    url_for_static_asset,
)

try:
    from mako.exceptions import TemplateLookupException as MakoTemplateNotFound  # type: ignore[import-untyped]
    from mako.lookup import TemplateLookup  # type: ignore[import-untyped]
except ImportError as e:
    raise MissingDependencyException("mako") from e

if TYPE_CHECKING:
    from pathlib import Path

    from mako.template import Template as _MakoTemplate  # type: ignore[import-untyped]

__all__ = ("MakoTemplate", "MakoTemplateEngine")

P = ParamSpec("P")
T = TypeVar("T")


class MakoTemplate(TemplateProtocol):
    """Mako template, implementing ``TemplateProtocol``"""

    def __init__(self, template: _MakoTemplate, template_callables: list[tuple[str, TemplateCallableType]]) -> None:
        """Initialize a template.

        Args:
            template: Base ``MakoTemplate`` used by the underlying mako-engine
            template_callables: List of callables passed to the template
        """
        super().__init__()
        self.template = template
        self.template_callables = template_callables

    def render(self, *args: Any, **kwargs: Any) -> str:
        """Render a template.

        Args:
            args: Positional arguments passed to the engines ``render`` function
            kwargs: Keyword arguments passed to the engines ``render`` function

        Returns:
            Rendered template as a string
        """
        for callable_key, template_callable in self.template_callables:
            kwargs_copy = {**kwargs}
            kwargs[callable_key] = partial(template_callable, kwargs_copy)

        return str(self.template.render(*args, **kwargs))


class MakoTemplateEngine(TemplateEngineProtocol[MakoTemplate, Mapping[str, Any]]):
    """Mako based TemplateEngine."""

    def __init__(self, directory: Path | list[Path] | None = None, engine_instance: Any | None = None) -> None:
        """Initialize template engine.

        Args:
            directory: Direct path or list of directory paths from which to serve templates.
            engine_instance: A mako TemplateLookup instance.
        """
        super().__init__(directory, engine_instance)
        if directory and engine_instance:
            raise ImproperlyConfiguredException("You must provide either a directory or a mako TemplateLookup.")
        if directory:
            self.engine = TemplateLookup(
                directories=directory if isinstance(directory, (list, tuple)) else [directory], default_filters=["h"]
            )
        elif engine_instance:
            self.engine = engine_instance

        self._template_callables: list[tuple[str, TemplateCallableType]] = []
        self.register_template_callable(key="url_for_static_asset", template_callable=url_for_static_asset)
        self.register_template_callable(key="csrf_token", template_callable=csrf_token)
        self.register_template_callable(key="url_for", template_callable=url_for)

    def get_template(self, template_name: str) -> MakoTemplate:
        """Retrieve a template by matching its name (dotted path) with files in the directory or directories provided.

        Args:
            template_name: A dotted path

        Returns:
            MakoTemplate instance

        Raises:
            TemplateNotFoundException: if no template is found.
        """
        try:
            return MakoTemplate(
                template=self.engine.get_template(template_name), template_callables=self._template_callables
            )
        except MakoTemplateNotFound as exc:
            raise TemplateNotFoundException(template_name=template_name) from exc

    def register_template_callable(
        self, key: str, template_callable: TemplateCallableType[Mapping[str, Any], P, T]
    ) -> None:
        """Register a callable on the template engine.

        Args:
            key: The callable key, i.e. the value to use inside the template to call the callable.
            template_callable: A callable to register.

        Returns:
            None
        """
        self._template_callables.append((key, template_callable))

    @classmethod
    def from_template_lookup(cls, template_lookup: TemplateLookup) -> MakoTemplateEngine:
        """Create a template engine from an existing mako TemplateLookup instance.

        Args:
            template_lookup: A mako TemplateLookup instance.

        Returns:
            MakoTemplateEngine instance
        """
        return cls(directory=None, engine_instance=template_lookup)
