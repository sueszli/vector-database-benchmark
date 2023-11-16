import html
from contextlib import closing
from inspect import isclass
from io import StringIO
from pathlib import Path
from string import Template
from .. import __version__, config_context
from .fixes import parse_version

class _IDCounter:
    """Generate sequential ids with a prefix."""

    def __init__(self, prefix):
        if False:
            return 10
        self.prefix = prefix
        self.count = 0

    def get_id(self):
        if False:
            print('Hello World!')
        self.count += 1
        return f'{self.prefix}-{self.count}'

def _get_css_style():
    if False:
        while True:
            i = 10
    return Path(__file__).with_suffix('.css').read_text(encoding='utf-8')
_CONTAINER_ID_COUNTER = _IDCounter('sk-container-id')
_ESTIMATOR_ID_COUNTER = _IDCounter('sk-estimator-id')
_CSS_STYLE = _get_css_style()

class _VisualBlock:
    """HTML Representation of Estimator

    Parameters
    ----------
    kind : {'serial', 'parallel', 'single'}
        kind of HTML block

    estimators : list of estimators or `_VisualBlock`s or a single estimator
        If kind != 'single', then `estimators` is a list of
        estimators.
        If kind == 'single', then `estimators` is a single estimator.

    names : list of str, default=None
        If kind != 'single', then `names` corresponds to estimators.
        If kind == 'single', then `names` is a single string corresponding to
        the single estimator.

    name_details : list of str, str, or None, default=None
        If kind != 'single', then `name_details` corresponds to `names`.
        If kind == 'single', then `name_details` is a single string
        corresponding to the single estimator.

    dash_wrapped : bool, default=True
        If true, wrapped HTML element will be wrapped with a dashed border.
        Only active when kind != 'single'.
    """

    def __init__(self, kind, estimators, *, names=None, name_details=None, dash_wrapped=True):
        if False:
            for i in range(10):
                print('nop')
        self.kind = kind
        self.estimators = estimators
        self.dash_wrapped = dash_wrapped
        if self.kind in ('parallel', 'serial'):
            if names is None:
                names = (None,) * len(estimators)
            if name_details is None:
                name_details = (None,) * len(estimators)
        self.names = names
        self.name_details = name_details

    def _sk_visual_block_(self):
        if False:
            i = 10
            return i + 15
        return self

def _write_label_html(out, name, name_details, outer_class='sk-label-container', inner_class='sk-label', checked=False, doc_link='', is_fitted_css_class='', is_fitted_icon=''):
    if False:
        i = 10
        return i + 15
    'Write labeled html with or without a dropdown with named details.\n\n    Parameters\n    ----------\n    out : file-like object\n        The file to write the HTML representation to.\n    name : str\n        The label for the estimator. It corresponds either to the estimator class name\n        for a simple estimator or in the case of a `Pipeline` and `ColumnTransformer`,\n        it corresponds to the name of the step.\n    name_details : str\n        The details to show as content in the dropdown part of the toggleable label. It\n        can contain information such as non-default parameters or column information for\n        `ColumnTransformer`.\n    outer_class : {"sk-label-container", "sk-item"}, default="sk-label-container"\n        The CSS class for the outer container.\n    inner_class : {"sk-label", "sk-estimator"}, default="sk-label"\n        The CSS class for the inner container.\n    checked : bool, default=False\n        Whether the dropdown is folded or not. With a single estimator, we intend to\n        unfold the content.\n    doc_link : str, default=""\n        The link to the documentation for the estimator. If an empty string, no link is\n        added to the diagram. This can be generated for an estimator if it uses the\n        `_HTMLDocumentationLinkMixin`.\n    is_fitted_css_class : {"", "fitted"}\n        The CSS class to indicate whether or not the estimator is fitted. The\n        empty string means that the estimator is not fitted and "fitted" means that the\n        estimator is fitted.\n    is_fitted_icon : str, default=""\n        The HTML representation to show the fitted information in the diagram. An empty\n        string means that no information is shown.\n    '
    padding_label = '&nbsp;' if is_fitted_icon else ''
    out.write(f'<div class="{outer_class}"><div class="{inner_class} {is_fitted_css_class} sk-toggleable">')
    name = html.escape(name)
    if name_details is not None:
        name_details = html.escape(str(name_details))
        label_class = f'sk-toggleable__label {is_fitted_css_class} sk-toggleable__label-arrow'
        checked_str = 'checked' if checked else ''
        est_id = _ESTIMATOR_ID_COUNTER.get_id()
        if doc_link:
            doc_label = '<span>Online documentation</span>'
            if name is not None:
                doc_label = f'<span>Documentation for {name}</span>'
            doc_link = f'<a class="sk-estimator-doc-link {is_fitted_css_class}" rel="noreferrer" target="_blank" href="{doc_link}">?{doc_label}</a>'
            padding_label += '&nbsp;'
        fmt_str = f'<input class="sk-toggleable__control sk-hidden--visually" id="{est_id}" type="checkbox" {checked_str}><label for="{est_id}" class="{label_class} {is_fitted_css_class}">{padding_label}{name}{doc_link}{is_fitted_icon}</label><div class="sk-toggleable__content {is_fitted_css_class}"><pre>{name_details}</pre></div> '
        out.write(fmt_str)
    else:
        out.write(f'<label>{name}</label>')
    out.write('</div></div>')

def _get_visual_block(estimator):
    if False:
        print('Hello World!')
    'Generate information about how to display an estimator.'
    if hasattr(estimator, '_sk_visual_block_'):
        try:
            return estimator._sk_visual_block_()
        except Exception:
            return _VisualBlock('single', estimator, names=estimator.__class__.__name__, name_details=str(estimator))
    if isinstance(estimator, str):
        return _VisualBlock('single', estimator, names=estimator, name_details=estimator)
    elif estimator is None:
        return _VisualBlock('single', estimator, names='None', name_details='None')
    if hasattr(estimator, 'get_params') and (not isclass(estimator)):
        estimators = [(key, est) for (key, est) in estimator.get_params(deep=False).items() if hasattr(est, 'get_params') and hasattr(est, 'fit') and (not isclass(est))]
        if estimators:
            return _VisualBlock('parallel', [est for (_, est) in estimators], names=[f'{key}: {est.__class__.__name__}' for (key, est) in estimators], name_details=[str(est) for (_, est) in estimators])
    return _VisualBlock('single', estimator, names=estimator.__class__.__name__, name_details=str(estimator))

def _write_estimator_html(out, estimator, estimator_label, estimator_label_details, is_fitted_css_class, is_fitted_icon='', first_call=False):
    if False:
        while True:
            i = 10
    'Write estimator to html in serial, parallel, or by itself (single).\n\n    For multiple estimators, this function is called recursively.\n\n    Parameters\n    ----------\n    out : file-like object\n        The file to write the HTML representation to.\n    estimator : estimator object\n        The estimator to visualize.\n    estimator_label : str\n        The label for the estimator. It corresponds either to the estimator class name\n        for simple estimator or in the case of `Pipeline` and `ColumnTransformer`, it\n        corresponds to the name of the step.\n    estimator_label_details : str\n        The details to show as content in the dropdown part of the toggleable label.\n        It can contain information as non-default parameters or column information for\n        `ColumnTransformer`.\n    is_fitted_css_class : {"", "fitted"}\n        The CSS class to indicate whether or not the estimator is fitted or not. The\n        empty string means that the estimator is not fitted and "fitted" means that the\n        estimator is fitted.\n    is_fitted_icon : str, default=""\n        The HTML representation to show the fitted information in the diagram. An empty\n        string means that no information is shown. If the estimator to be shown is not\n        the first estimator (i.e. `first_call=False`), `is_fitted_icon` is always an\n        empty string.\n    first_call : bool, default=False\n        Whether this is the first time this function is called.\n    '
    if first_call:
        est_block = _get_visual_block(estimator)
    else:
        is_fitted_icon = ''
        with config_context(print_changed_only=True):
            est_block = _get_visual_block(estimator)
    if hasattr(estimator, '_get_doc_link'):
        doc_link = estimator._get_doc_link()
    else:
        doc_link = ''
    if est_block.kind in ('serial', 'parallel'):
        dashed_wrapped = first_call or est_block.dash_wrapped
        dash_cls = ' sk-dashed-wrapped' if dashed_wrapped else ''
        out.write(f'<div class="sk-item{dash_cls}">')
        if estimator_label:
            _write_label_html(out, estimator_label, estimator_label_details, doc_link=doc_link, is_fitted_css_class=is_fitted_css_class, is_fitted_icon=is_fitted_icon)
        kind = est_block.kind
        out.write(f'<div class="sk-{kind}">')
        est_infos = zip(est_block.estimators, est_block.names, est_block.name_details)
        for (est, name, name_details) in est_infos:
            if kind == 'serial':
                _write_estimator_html(out, est, name, name_details, is_fitted_css_class=is_fitted_css_class)
            else:
                out.write('<div class="sk-parallel-item">')
                serial_block = _VisualBlock('serial', [est], dash_wrapped=False)
                _write_estimator_html(out, serial_block, name, name_details, is_fitted_css_class=is_fitted_css_class)
                out.write('</div>')
        out.write('</div></div>')
    elif est_block.kind == 'single':
        _write_label_html(out, est_block.names, est_block.name_details, outer_class='sk-item', inner_class='sk-estimator', checked=first_call, doc_link=doc_link, is_fitted_css_class=is_fitted_css_class, is_fitted_icon=is_fitted_icon)

def estimator_html_repr(estimator):
    if False:
        i = 10
        return i + 15
    'Build a HTML representation of an estimator.\n\n    Read more in the :ref:`User Guide <visualizing_composite_estimators>`.\n\n    Parameters\n    ----------\n    estimator : estimator object\n        The estimator to visualize.\n\n    Returns\n    -------\n    html: str\n        HTML representation of estimator.\n    '
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted
    if not hasattr(estimator, 'fit'):
        status_label = '<span>Not fitted</span>'
        is_fitted_css_class = ''
    else:
        try:
            check_is_fitted(estimator)
            status_label = '<span>Fitted</span>'
            is_fitted_css_class = 'fitted'
        except NotFittedError:
            status_label = '<span>Not fitted</span>'
            is_fitted_css_class = ''
    is_fitted_icon = f'<span class="sk-estimator-doc-link {is_fitted_css_class}">i{status_label}</span>'
    with closing(StringIO()) as out:
        container_id = _CONTAINER_ID_COUNTER.get_id()
        style_template = Template(_CSS_STYLE)
        style_with_id = style_template.substitute(id=container_id)
        estimator_str = str(estimator)
        fallback_msg = 'In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.'
        html_template = f'<style>{style_with_id}</style><div id="{container_id}" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>{html.escape(estimator_str)}</pre><b>{fallback_msg}</b></div><div class="sk-container" hidden>'
        out.write(html_template)
        _write_estimator_html(out, estimator, estimator.__class__.__name__, estimator_str, first_call=True, is_fitted_css_class=is_fitted_css_class, is_fitted_icon=is_fitted_icon)
        out.write('</div></div>')
        html_output = out.getvalue()
        return html_output

class _HTMLDocumentationLinkMixin:
    """Mixin class allowing to generate a link to the API documentation.

    This mixin relies on three attributes:
    - `_doc_link_module`: it corresponds to the root module (e.g. `sklearn`). Using this
      mixin, the default value is `sklearn`.
    - `_doc_link_template`: it corresponds to the template used to generate the
      link to the API documentation. Using this mixin, the default value is
      `"https://scikit-learn.org/{version_url}/modules/generated/
      {estimator_module}.{estimator_name}.html"`.
    - `_doc_link_url_param_generator`: it corresponds to a function that generates the
      parameters to be used in the template when the estimator module and name are not
      sufficient.

    The method :meth:`_get_doc_link` generates the link to the API documentation for a
    given estimator.

    This useful provides all the necessary states for
    :func:`sklearn.utils.estimator_html_repr` to generate a link to the API
    documentation for the estimator HTML diagram.

    Examples
    --------
    If the default values for `_doc_link_module`, `_doc_link_template` are not suitable,
    then you can override them:
    >>> from sklearn.base import BaseEstimator
    >>> estimator = BaseEstimator()
    >>> estimator._doc_link_template = "https://website.com/{single_param}.html"
    >>> def url_param_generator(estimator):
    ...     return {"single_param": estimator.__class__.__name__}
    >>> estimator._doc_link_url_param_generator = url_param_generator
    >>> estimator._get_doc_link()
    'https://website.com/BaseEstimator.html'
    """
    _doc_link_module = 'sklearn'
    _doc_link_url_param_generator = None

    @property
    def _doc_link_template(self):
        if False:
            while True:
                i = 10
        sklearn_version = parse_version(__version__)
        if sklearn_version.dev is None:
            version_url = f'{sklearn_version.major}.{sklearn_version.minor}'
        else:
            version_url = 'dev'
        return getattr(self, '__doc_link_template', f'https://scikit-learn.org/{version_url}/modules/generated/{{estimator_module}}.{{estimator_name}}.html')

    @_doc_link_template.setter
    def _doc_link_template(self, value):
        if False:
            while True:
                i = 10
        setattr(self, '__doc_link_template', value)

    def _get_doc_link(self):
        if False:
            for i in range(10):
                print('nop')
        'Generates a link to the API documentation for a given estimator.\n\n        This method generates the link to the estimator\'s documentation page\n        by using the template defined by the attribute `_doc_link_template`.\n\n        Returns\n        -------\n        url : str\n            The URL to the API documentation for this estimator. If the estimator does\n            not belong to module `_doc_link_module`, the empty string (i.e. `""`) is\n            returned.\n        '
        if self.__class__.__module__.split('.')[0] != self._doc_link_module:
            return ''
        if self._doc_link_url_param_generator is None:
            estimator_name = self.__class__.__name__
            estimator_module = '.'.join([_ for _ in self.__class__.__module__.split('.') if not _.startswith('_')])
            return self._doc_link_template.format(estimator_module=estimator_module, estimator_name=estimator_name)
        return self._doc_link_template.format(**self._doc_link_url_param_generator(self))