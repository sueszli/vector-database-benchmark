""" Provide a base class for all objects (called Bokeh Models) that can go in
a Bokeh |Document|.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from operator import itemgetter
from typing import TYPE_CHECKING, Any
from ..util.serialization import make_id
from ..util.strings import append_docstring
if TYPE_CHECKING:
    from .model import Model
__all__ = ('html_repr', 'process_example')

def html_repr(obj: Model):
    if False:
        print('Hello World!')
    module = obj.__class__.__module__
    name = obj.__class__.__name__
    _id = getattr(obj, '_id', None)
    cls_name = make_id()

    def row(c: str):
        if False:
            i = 10
            return i + 15
        return f'<div style="display: table-row;">{c}</div>'

    def hidden_row(c: str):
        if False:
            for i in range(10):
                print('nop')
        return f'<div class="{cls_name}" style="display: none;">{c}</div>'

    def cell(c: str):
        if False:
            while True:
                i = 10
        return f'<div style="display: table-cell;">{c}</div>'
    html = ''
    html += '<div style="display: table;">'
    ellipsis_id = make_id()
    ellipsis = f'<span id="{ellipsis_id}" style="cursor: pointer;">&hellip;)</span>'
    prefix = cell(f'<b title="{module}.{name}">{name}</b>(')
    html += row(prefix + cell('id' + '&nbsp;=&nbsp;' + repr(_id) + ', ' + ellipsis))
    props = obj.properties_with_values().items()
    sorted_props = sorted(props, key=itemgetter(0))
    all_props = sorted_props
    for (i, (prop, value)) in enumerate(all_props):
        end = ')' if i == len(all_props) - 1 else ','
        html += hidden_row(cell('') + cell(prop + '&nbsp;=&nbsp;' + repr(value) + end))
    html += '</div>'
    html += _HTML_REPR % dict(ellipsis_id=ellipsis_id, cls_name=cls_name)
    return html

def process_example(cls: type[Any]) -> None:
    if False:
        i = 10
        return i + 15
    ' A decorator to mark abstract base classes derived from |HasProps|.\n\n    '
    if '__example__' in cls.__dict__:
        cls.__doc__ = append_docstring(cls.__doc__, _EXAMPLE_TEMPLATE.format(path=cls.__dict__['__example__']))
_HTML_REPR = '\n<script>\n(function() {\n  let expanded = false;\n  const ellipsis = document.getElementById("%(ellipsis_id)s");\n  ellipsis.addEventListener("click", function() {\n    const rows = document.getElementsByClassName("%(cls_name)s");\n    for (let i = 0; i < rows.length; i++) {\n      const el = rows[i];\n      el.style.display = expanded ? "none" : "table-row";\n    }\n    ellipsis.innerHTML = expanded ? "&hellip;)" : "&lsaquo;&lsaquo;&lsaquo;";\n    expanded = !expanded;\n  });\n})();\n</script>\n'
_EXAMPLE_TEMPLATE = '\n\n    Example\n    -------\n\n    .. bokeh-plot:: __REPO__/{path}\n        :source-position: below\n\n'