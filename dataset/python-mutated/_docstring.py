from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from inspect import Parameter
from ..models import Marker
__all__ = ('generate_docstring',)

def generate_docstring(glyphclass, parameters, extra_docs):
    if False:
        return 10
    return f' {_docstring_header(glyphclass)}\n\nArgs:\n{_docstring_args(parameters)}\n\nKeyword args:\n{_docstring_kwargs(parameters)}\n\n{_docstring_other()}\n\nIt is also possible to set the color and alpha parameters of extra glyphs for\nselection, nonselection, hover, or muted. To do so, add the relevant prefix to\nany visual parameter. For example, pass ``nonselection_alpha`` to set the line\nand fill alpha for nonselect, or ``hover_fill_alpha`` to set the fill alpha for\nhover. See the :ref:`ug_styling_plots_glyphs` section of the user guide for\nfull details.\n\nReturns:\n    :class:`~bokeh.models.renderers.GlyphRenderer`\n\n{_docstring_extra(extra_docs)}\n'

def _add_arglines(arglines, param, typ, doc):
    if False:
        return 10
    default = param.default if param.default != Parameter.empty else None
    arglines.append(f"    {param.name} ({typ}{(', optional' if default else '')}):")
    if doc:
        arglines += [f'    {x}' for x in doc.rstrip().strip('\n').split('\n')]
    if arglines and default is not None:
        arglines[-1] += f' (default: {default!r})'
    arglines.append('')

def _docstring_args(parameters):
    if False:
        return 10
    arglines = []
    for (param, typ, doc) in (x for x in parameters if x[0].kind == Parameter.POSITIONAL_OR_KEYWORD):
        _add_arglines(arglines, param, typ, doc)
    return '\n'.join(arglines)

def _docstring_extra(extra_docs):
    if False:
        print('Hello World!')
    return '' if extra_docs is None else extra_docs

def _docstring_header(glyphclass):
    if False:
        i = 10
        return i + 15
    glyph_class = 'Scatter' if issubclass(glyphclass, Marker) else glyphclass.__name__
    return f'Configure and add :class:`~bokeh.models.glyphs.{glyph_class}` glyphs to this figure.'

def _docstring_kwargs(parameters):
    if False:
        print('Hello World!')
    arglines = []
    for (param, typ, doc) in (x for x in parameters if x[0].kind == Parameter.KEYWORD_ONLY):
        _add_arglines(arglines, param, typ, doc)
    return '\n'.join(arglines)

def _docstring_other():
    if False:
        return 10
    return _OTHER_PARAMS
_OTHER_PARAMS = '\nOther Parameters:\n    alpha (float, optional) :\n        An alias to set all alpha keyword arguments at once. (default: None)\n\n        Alpha values must be between 0 (fully transparent) and 1 (fully opaque).\n\n        Any explicitly set values for ``line_alpha``, etc. will override this\n        setting.\n\n    color (color, optional) :\n        An alias to set all color keyword arguments at once. (default: None)\n\n        See :ref:`ug_styling_colors` in the user guide for different\n        options to define colors.\n\n        Any explicitly set values for ``line_color``, etc. will override this\n        setting.\n\n    legend_field (str, optional) :\n        Specify that the glyph should produce multiple legend entries by\n        :ref:`grouping them in the browser <ug_basic_annotations_legends_legend_field>`.\n        The value of this parameter is the name of a column in the data source\n        that should be used for the grouping.\n\n        The grouping is performed *in JavaScript*, at the same time the Bokeh\n        content is rendered in the browser. If the data is subsequently updated,\n        the legend will automatically re-group.\n\n        .. note::\n            Only one of ``legend_field``, ``legend_group``, or ``legend_label``\n            should be supplied\n\n    legend_group (str, optional) :\n        Specify that the glyph should produce multiple legend entries by\n        :ref:`grouping them in Python <ug_basic_annotations_legends_legend_group>`.\n        The value of this parameter is the name of a column in the data source\n        that should be used for the grouping.\n\n        The grouping is performed in Python, before the Bokeh output is sent to\n        a browser. If the date is subsequently updated, the legend will *not*\n        automatically re-group.\n\n        .. note::\n            Only one of ``legend_field``, ``legend_group``, or ``legend_label``\n            should be supplied\n\n    legend_label (str, optional) :\n        Specify that the glyph should produce a single\n        :ref:`basic legend label <ug_basic_annotations_legends_legend_label>` in\n        the legend. The legend entry is labeled with the exact text supplied\n        here.\n\n        .. note::\n            Only one of ``legend_field``, ``legend_group``, or ``legend_label``\n            should be supplied.\n\n    muted (bool, optionall) :\n        Whether the glyph should be rendered as muted (default: False)\n\n        For this to be useful, an ``muted_glyph`` must be configured on the\n        returned ``GlyphRender``. This can be done by explicitly creating a\n        ``Glyph`` to use, or more simply by passing e.g. ``muted_color``, etc.\n        to this glyph function.\n\n    name (str, optional) :\n        An optional user-supplied name to attach to the renderer (default: None)\n\n        Bokeh does not use this value in any way, but it may be useful for\n        searching a Bokeh document to find a specific model.\n\n    source (ColumnDataSource, optional) :\n        A user-supplied data source. (defatult: None)\n\n        If not supplied, Bokeh will automatically construct an internal\n        ``ColumnDataSource`` with default column names from the coordinates and\n        other arguments that were passed-in as literal list or array values.\n\n        If supplied, Bokeh will use the supplied data source to derive the glyph.\n        In this case, literal list or arrays may not be used for coordinates or\n        other arguments. Only singular fixed values (e.g. ``x=10``) or column\n        names in the data source (e.g. ``x="time"``) are permitted.\n\n    view (CDSView, optional) :\n        A view for filtering the data source. (default: None)\n\n    visible (bool, optional) :\n        Whether the glyph should be rendered. (default: True)\n\n    x_range_name (str, optional) :\n        The name of an extra range to use for mapping x-coordinates.\n        (default: None)\n\n        If not supplied, then the default ``y_range`` of the plot will be used\n        for coordinate mapping.\n\n    y_range_name (str, optional) :\n        The name of an extra range to use for mapping y-coordinates.\n        (default: None)\n\n        If not supplied, then the default ``y_range`` of the plot will be used\n        for coordinate mapping.\n\n    level (RenderLevel, optional) :\n        Specify the render level order for this glyph.\n\n'