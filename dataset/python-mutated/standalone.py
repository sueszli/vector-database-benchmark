"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypedDict, Union, cast, overload
from .. import __version__
from ..core.templates import AUTOLOAD_JS, AUTOLOAD_TAG, FILE, MACROS, ROOT_DIV
from ..document.document import DEFAULT_TITLE, Document
from ..model import Model
from ..resources import Resources, ResourcesLike
from ..themes import Theme
from .bundle import Script, bundle_for_objs_and_resources
from .elements import html_page_for_render_items, script_for_render_items
from .util import FromCurdoc, OutputDocumentFor, RenderRoot, standalone_docs_json, standalone_docs_json_and_render_items
from .wrappers import wrap_in_onload
if TYPE_CHECKING:
    from jinja2 import Template
    from typing_extensions import TypeAlias
    from ..core.types import ID
    from ..document.document import DocJson
__all__ = ('autoload_static', 'components', 'file_html', 'json_item')
ModelLike: TypeAlias = Union[Model, Document]
ModelLikeCollection: TypeAlias = Union[Sequence[ModelLike], dict[str, ModelLike]]
ThemeLike: TypeAlias = Union[None, Theme, type[FromCurdoc]]

def autoload_static(model: Model | Document, resources: Resources, script_path: str) -> tuple[str, str]:
    if False:
        print('Hello World!')
    ' Return JavaScript code and a script tag that can be used to embed\n    Bokeh Plots.\n\n    The data for the plot is stored directly in the returned JavaScript code.\n\n    Args:\n        model (Model or Document) :\n\n        resources (Resources) :\n\n        script_path (str) :\n\n    Returns:\n        (js, tag) :\n            JavaScript code to be saved at ``script_path`` and a ``<script>``\n            tag to load it\n\n    Raises:\n        ValueError\n\n    '
    if isinstance(model, Model):
        models = [model]
    elif isinstance(model, Document):
        models = model.roots
    else:
        raise ValueError('autoload_static expects a single Model or Document')
    with OutputDocumentFor(models):
        (docs_json, [render_item]) = standalone_docs_json_and_render_items([model])
    bundle = bundle_for_objs_and_resources(None, resources)
    bundle.add(Script(script_for_render_items(docs_json, [render_item])))
    (_, elementid) = next(iter(render_item.roots.to_json().items()))
    js = wrap_in_onload(AUTOLOAD_JS.render(bundle=bundle, elementid=elementid))
    tag = AUTOLOAD_TAG.render(src_path=script_path, elementid=elementid)
    return (js, tag)

@overload
def components(models: Model, wrap_script: bool=..., wrap_plot_info: Literal[True]=..., theme: ThemeLike=...) -> tuple[str, str]:
    if False:
        i = 10
        return i + 15
    ...

@overload
def components(models: Model, wrap_script: bool=..., wrap_plot_info: Literal[False]=..., theme: ThemeLike=...) -> tuple[str, RenderRoot]:
    if False:
        while True:
            i = 10
    ...

@overload
def components(models: Sequence[Model], wrap_script: bool=..., wrap_plot_info: Literal[True]=..., theme: ThemeLike=...) -> tuple[str, Sequence[str]]:
    if False:
        print('Hello World!')
    ...

@overload
def components(models: Sequence[Model], wrap_script: bool=..., wrap_plot_info: Literal[False]=..., theme: ThemeLike=...) -> tuple[str, Sequence[RenderRoot]]:
    if False:
        print('Hello World!')
    ...

@overload
def components(models: dict[str, Model], wrap_script: bool=..., wrap_plot_info: Literal[True]=..., theme: ThemeLike=...) -> tuple[str, dict[str, str]]:
    if False:
        i = 10
        return i + 15
    ...

@overload
def components(models: dict[str, Model], wrap_script: bool=..., wrap_plot_info: Literal[False]=..., theme: ThemeLike=...) -> tuple[str, dict[str, RenderRoot]]:
    if False:
        while True:
            i = 10
    ...

def components(models: Model | Sequence[Model] | dict[str, Model], wrap_script: bool=True, wrap_plot_info: bool=True, theme: ThemeLike=None) -> tuple[str, Any]:
    if False:
        print('Hello World!')
    ' Return HTML components to embed a Bokeh plot. The data for the plot is\n    stored directly in the returned HTML.\n\n    An example can be found in examples/embed/embed_multiple.py\n\n    The returned components assume that BokehJS resources are **already loaded**.\n    The HTML document or template in which they will be embedded needs to\n    include scripts tags, either from a local URL or Bokeh\'s CDN (replacing\n    ``x.y.z`` with the version of Bokeh you are using):\n\n    .. code-block:: html\n\n        <script src="https://cdn.bokeh.org/bokeh/release/bokeh-x.y.z.min.js"></script>\n        <script src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-x.y.z.min.js"></script>\n        <script src="https://cdn.bokeh.org/bokeh/release/bokeh-tables-x.y.z.min.js"></script>\n        <script src="https://cdn.bokeh.org/bokeh/release/bokeh-gl-x.y.z.min.js"></script>\n        <script src="https://cdn.bokeh.org/bokeh/release/bokeh-mathjax-x.y.z.min.js"></script>\n\n    Only the Bokeh core library ``bokeh-x.y.z.min.js`` is always required. The\n    other scripts are optional and only need to be included if you want to use\n    corresponding features:\n\n    * The ``"bokeh-widgets"`` files are only necessary if you are using any of the\n      :ref:`Bokeh widgets <ug_interaction_widgets>`.\n    * The ``"bokeh-tables"`` files are only necessary if you are using Bokeh\'s\n      :ref:`data tables <ug_interaction_widgets_examples_datatable>`.\n    * The ``"bokeh-api"`` files are required to use the\n      :ref:`BokehJS API <ug_advanced_bokehjs>` and must be loaded *after* the\n      core BokehJS library.\n    * The ``"bokeh-gl"`` files are required to enable\n      :ref:`WebGL support <ug_output_webgl>`.\n    * the ``"bokeh-mathjax"`` files are required to enable\n      :ref:`MathJax support <ug_styling_mathtext>`.\n\n    Args:\n        models (Model|list|dict|tuple) :\n            A single Model, a list/tuple of Models, or a dictionary of keys\n            and Models.\n\n        wrap_script (boolean, optional) :\n            If True, the returned javascript is wrapped in a script tag.\n            (default: True)\n\n        wrap_plot_info (boolean, optional) :\n            If True, returns ``<div>`` strings. Otherwise, return\n            :class:`~bokeh.embed.RenderRoot` objects that can be used to build\n            your own divs. (default: True)\n\n        theme (Theme, optional) :\n            Applies the specified theme when creating the components. If None,\n            or not specified, and the supplied models constitute the full set\n            of roots of a document, applies the theme of that document to the\n            components. Otherwise applies the default theme.\n\n    Returns:\n        UTF-8 encoded *(script, div[s])* or *(raw_script, plot_info[s])*\n\n    Examples:\n\n        With default wrapping parameter values:\n\n        .. code-block:: python\n\n            components(plot)\n            # => (script, plot_div)\n\n            components((plot1, plot2))\n            # => (script, (plot1_div, plot2_div))\n\n            components({"Plot 1": plot1, "Plot 2": plot2})\n            # => (script, {"Plot 1": plot1_div, "Plot 2": plot2_div})\n\n    Examples:\n\n        With wrapping parameters set to ``False``:\n\n        .. code-block:: python\n\n            components(plot, wrap_script=False, wrap_plot_info=False)\n            # => (javascript, plot_root)\n\n            components((plot1, plot2), wrap_script=False, wrap_plot_info=False)\n            # => (javascript, (plot1_root, plot2_root))\n\n            components({"Plot 1": plot1, "Plot 2": plot2}, wrap_script=False, wrap_plot_info=False)\n            # => (javascript, {"Plot 1": plot1_root, "Plot 2": plot2_root})\n\n    '
    was_single_object = False
    if isinstance(models, Model):
        was_single_object = True
        models = [models]
    models = _check_models_or_docs(models)
    model_keys = None
    dict_type: type[dict[Any, Any]] = dict
    if isinstance(models, dict):
        dict_type = models.__class__
        model_keys = models.keys()
        models = list(models.values())
    with OutputDocumentFor(models, apply_theme=theme):
        (docs_json, [render_item]) = standalone_docs_json_and_render_items(models)
    bundle = bundle_for_objs_and_resources(None, None)
    bundle.add(Script(script_for_render_items(docs_json, [render_item])))
    script = bundle.scripts(tag=wrap_script)

    def div_for_root(root: RenderRoot) -> str:
        if False:
            while True:
                i = 10
        return ROOT_DIV.render(root=root, macros=MACROS)
    results: list[str] | list[RenderRoot]
    if wrap_plot_info:
        results = [div_for_root(root) for root in render_item.roots]
    else:
        results = list(render_item.roots)
    result: Any
    if was_single_object:
        result = results[0]
    elif model_keys is not None:
        result = dict_type(zip(model_keys, results))
    else:
        result = tuple(results)
    return (script, result)

def file_html(models: Model | Document | Sequence[Model], resources: ResourcesLike | None=None, title: str | None=None, *, template: Template | str=FILE, template_variables: dict[str, Any]={}, theme: ThemeLike=None, suppress_callback_warning: bool=False, _always_new: bool=False) -> str:
    if False:
        for i in range(10):
            print('nop')
    ' Return an HTML document that embeds Bokeh Model or Document objects.\n\n    The data for the plot is stored directly in the returned HTML, with\n    support for customizing the JS/CSS resources independently and\n    customizing the jinja2 template.\n\n    Args:\n        models (Model or Document or seq[Model]) : Bokeh object or objects to render\n            typically a Model or Document\n\n        resources (ResourcesLike) :\n            A resources configuration for Bokeh JS & CSS assets.\n\n        title (str, optional) :\n            A title for the HTML document ``<title>`` tags or None. (default: None)\n\n            If None, attempt to automatically find the Document title from the given\n            plot objects.\n\n        template (Template, optional) : HTML document template (default: FILE)\n            A Jinja2 Template, see bokeh.core.templates.FILE for the required\n            template parameters\n\n        template_variables (dict, optional) : variables to be used in the Jinja2\n            template. If used, the following variable names will be overwritten:\n            title, bokeh_js, bokeh_css, plot_script, plot_div\n\n        theme (Theme, optional) :\n            Applies the specified theme to the created html. If ``None``, or\n            not specified, and the function is passed a document or the full set\n            of roots of a document, applies the theme of that document.  Otherwise\n            applies the default theme.\n\n        suppress_callback_warning (bool, optional) :\n            Normally generating standalone HTML from a Bokeh Document that has\n            Python callbacks will result in a warning stating that the callbacks\n            cannot function. However, this warning can be suppressed by setting\n            this value to True (default: False)\n\n    Returns:\n        UTF-8 encoded HTML\n\n    '
    models_seq: Sequence[Model] = []
    if isinstance(models, Model):
        models_seq = [models]
    elif isinstance(models, Document):
        if len(models.roots) == 0:
            raise ValueError('Document has no root Models')
        models_seq = models.roots
    else:
        models_seq = models
    resources = Resources.build(resources)
    with OutputDocumentFor(models_seq, apply_theme=theme, always_new=_always_new) as doc:
        (docs_json, render_items) = standalone_docs_json_and_render_items(models_seq, suppress_callback_warning=suppress_callback_warning)
        title = _title_from_models(models_seq, title)
        bundle = bundle_for_objs_and_resources([doc], resources)
        return html_page_for_render_items(bundle, docs_json, render_items, title=title, template=template, template_variables=template_variables)

class StandaloneEmbedJson(TypedDict):
    target_id: ID | None
    root_id: ID
    doc: DocJson
    version: str

def json_item(model: Model, target: ID | None=None, theme: ThemeLike=None) -> StandaloneEmbedJson:
    if False:
        print('Hello World!')
    ' Return a JSON block that can be used to embed standalone Bokeh content.\n\n    Args:\n        model (Model) :\n            The Bokeh object to embed\n\n        target (string, optional)\n            A div id to embed the model into. If None, the target id must\n            be supplied in the JavaScript call.\n\n        theme (Theme, optional) :\n            Applies the specified theme to the created html. If ``None``, or\n            not specified, and the function is passed a document or the full set\n            of roots of a document, applies the theme of that document.  Otherwise\n            applies the default theme.\n\n    Returns:\n        JSON-like\n\n    This function returns a JSON block that can be consumed by the BokehJS\n    function ``Bokeh.embed.embed_item``. As an example, a Flask endpoint for\n    ``/plot`` might return the following content to embed a Bokeh plot into\n    a div with id *"myplot"*:\n\n    .. code-block:: python\n\n        @app.route(\'/plot\')\n        def plot():\n            p = make_plot(\'petal_width\', \'petal_length\')\n            return json.dumps(json_item(p, "myplot"))\n\n    Then a web page can retrieve this JSON and embed the plot by calling\n    ``Bokeh.embed.embed_item``:\n\n    .. code-block:: html\n\n        <script>\n        fetch(\'/plot\')\n            .then(function(response) { return response.json(); })\n            .then(function(item) { Bokeh.embed.embed_item(item); })\n        </script>\n\n    Alternatively, if is more convenient to supply the target div id directly\n    in the page source, that is also possible. If `target_id` is omitted in the\n    call to this function:\n\n    .. code-block:: python\n\n        return json.dumps(json_item(p))\n\n    Then the value passed to ``embed_item`` is used:\n\n    .. code-block:: javascript\n\n        Bokeh.embed.embed_item(item, "myplot");\n\n    '
    with OutputDocumentFor([model], apply_theme=theme) as doc:
        doc.title = ''
        [doc_json] = standalone_docs_json([model]).values()
    root_id = doc_json['roots'][0]['id']
    return StandaloneEmbedJson(target_id=target, root_id=root_id, doc=doc_json, version=__version__)

def _check_models_or_docs(models: ModelLike | ModelLikeCollection) -> ModelLikeCollection:
    if False:
        for i in range(10):
            print('nop')
    '\n\n    '
    input_type_valid = False
    if isinstance(models, (Model, Document)):
        models = [models]
    if isinstance(models, Sequence) and all((isinstance(x, (Model, Document)) for x in models)):
        input_type_valid = True
    if isinstance(models, dict) and all((isinstance(x, str) for x in models.keys())) and all((isinstance(x, (Model, Document)) for x in models.values())):
        input_type_valid = True
    if not input_type_valid:
        raise ValueError('Input must be a Model, a Document, a Sequence of Models and Document, or a dictionary from string to Model and Document')
    return models

def _title_from_models(models: Sequence[Model | Document], title: str | None) -> str:
    if False:
        i = 10
        return i + 15
    if title is not None:
        return title
    for p in models:
        if isinstance(p, Document):
            return p.title
    for p in cast(Sequence[Model], models):
        if p.document is not None:
            return p.document.title
    return DEFAULT_TITLE