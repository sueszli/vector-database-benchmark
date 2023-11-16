from __future__ import annotations
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from xarray.core.formatting import inline_index_repr, inline_variable_array_repr, short_data_repr
from xarray.core.options import _get_boolean_with_default
STATIC_FILES = (('xarray.static.html', 'icons-svg-inline.html'), ('xarray.static.css', 'style.css'))

@lru_cache(None)
def _load_static_files():
    if False:
        for i in range(10):
            print('nop')
    'Lazily load the resource files into memory the first time they are needed'
    return [files(package).joinpath(resource).read_text(encoding='utf-8') for (package, resource) in STATIC_FILES]

def short_data_repr_html(array) -> str:
    if False:
        while True:
            i = 10
    'Format "data" for DataArray and Variable.'
    internal_data = getattr(array, 'variable', array)._data
    if hasattr(internal_data, '_repr_html_'):
        return internal_data._repr_html_()
    text = escape(short_data_repr(array))
    return f'<pre>{text}</pre>'

def format_dims(dims, dims_with_index) -> str:
    if False:
        for i in range(10):
            print('nop')
    if not dims:
        return ''
    dim_css_map = {dim: " class='xr-has-index'" if dim in dims_with_index else '' for dim in dims}
    dims_li = ''.join((f'<li><span{dim_css_map[dim]}>{escape(str(dim))}</span>: {size}</li>' for (dim, size) in dims.items()))
    return f"<ul class='xr-dim-list'>{dims_li}</ul>"

def summarize_attrs(attrs) -> str:
    if False:
        i = 10
        return i + 15
    attrs_dl = ''.join((f'<dt><span>{escape(str(k))} :</span></dt><dd>{escape(str(v))}</dd>' for (k, v) in attrs.items()))
    return f"<dl class='xr-attrs'>{attrs_dl}</dl>"

def _icon(icon_name) -> str:
    if False:
        while True:
            i = 10
    return f"<svg class='icon xr-{icon_name}'><use xlink:href='#{icon_name}'></use></svg>"

def summarize_variable(name, var, is_index=False, dtype=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    variable = var.variable if hasattr(var, 'variable') else var
    cssclass_idx = " class='xr-has-index'" if is_index else ''
    dims_str = f"({', '.join((escape(dim) for dim in var.dims))})"
    name = escape(str(name))
    dtype = dtype or escape(str(var.dtype))
    attrs_id = 'attrs-' + str(uuid.uuid4())
    data_id = 'data-' + str(uuid.uuid4())
    disabled = '' if len(var.attrs) else 'disabled'
    preview = escape(inline_variable_array_repr(variable, 35))
    attrs_ul = summarize_attrs(var.attrs)
    data_repr = short_data_repr_html(variable)
    attrs_icon = _icon('icon-file-text2')
    data_icon = _icon('icon-database')
    return f"<div class='xr-var-name'><span{cssclass_idx}>{name}</span></div><div class='xr-var-dims'>{dims_str}</div><div class='xr-var-dtype'>{dtype}</div><div class='xr-var-preview xr-preview'>{preview}</div><input id='{attrs_id}' class='xr-var-attrs-in' type='checkbox' {disabled}><label for='{attrs_id}' title='Show/Hide attributes'>{attrs_icon}</label><input id='{data_id}' class='xr-var-data-in' type='checkbox'><label for='{data_id}' title='Show/Hide data repr'>{data_icon}</label><div class='xr-var-attrs'>{attrs_ul}</div><div class='xr-var-data'>{data_repr}</div>"

def summarize_coords(variables) -> str:
    if False:
        for i in range(10):
            print('nop')
    li_items = []
    for (k, v) in variables.items():
        li_content = summarize_variable(k, v, is_index=k in variables.xindexes)
        li_items.append(f"<li class='xr-var-item'>{li_content}</li>")
    vars_li = ''.join(li_items)
    return f"<ul class='xr-var-list'>{vars_li}</ul>"

def summarize_vars(variables) -> str:
    if False:
        while True:
            i = 10
    vars_li = ''.join((f"<li class='xr-var-item'>{summarize_variable(k, v)}</li>" for (k, v) in variables.items()))
    return f"<ul class='xr-var-list'>{vars_li}</ul>"

def short_index_repr_html(index) -> str:
    if False:
        for i in range(10):
            print('nop')
    if hasattr(index, '_repr_html_'):
        return index._repr_html_()
    return f'<pre>{escape(repr(index))}</pre>'

def summarize_index(coord_names, index) -> str:
    if False:
        print('Hello World!')
    name = '<br>'.join([escape(str(n)) for n in coord_names])
    index_id = f'index-{uuid.uuid4()}'
    preview = escape(inline_index_repr(index))
    details = short_index_repr_html(index)
    data_icon = _icon('icon-database')
    return f"<div class='xr-index-name'><div>{name}</div></div><div class='xr-index-preview'>{preview}</div><div></div><input id='{index_id}' class='xr-index-data-in' type='checkbox'/><label for='{index_id}' title='Show/Hide index repr'>{data_icon}</label><div class='xr-index-data'>{details}</div>"

def summarize_indexes(indexes) -> str:
    if False:
        print('Hello World!')
    indexes_li = ''.join((f"<li class='xr-var-item'>{summarize_index(v, i)}</li>" for (v, i) in indexes.items()))
    return f"<ul class='xr-var-list'>{indexes_li}</ul>"

def collapsible_section(name, inline_details='', details='', n_items=None, enabled=True, collapsed=False) -> str:
    if False:
        i = 10
        return i + 15
    data_id = 'section-' + str(uuid.uuid4())
    has_items = n_items is not None and n_items
    n_items_span = '' if n_items is None else f' <span>({n_items})</span>'
    enabled = '' if enabled and has_items else 'disabled'
    collapsed = '' if collapsed or not has_items else 'checked'
    tip = " title='Expand/collapse section'" if enabled else ''
    return f"<input id='{data_id}' class='xr-section-summary-in' type='checkbox' {enabled} {collapsed}><label for='{data_id}' class='xr-section-summary' {tip}>{name}:{n_items_span}</label><div class='xr-section-inline-details'>{inline_details}</div><div class='xr-section-details'>{details}</div>"

def _mapping_section(mapping, name, details_func, max_items_collapse, expand_option_name, enabled=True) -> str:
    if False:
        for i in range(10):
            print('nop')
    n_items = len(mapping)
    expanded = _get_boolean_with_default(expand_option_name, n_items < max_items_collapse)
    collapsed = not expanded
    return collapsible_section(name, details=details_func(mapping), n_items=n_items, enabled=enabled, collapsed=collapsed)

def dim_section(obj) -> str:
    if False:
        i = 10
        return i + 15
    dim_list = format_dims(obj.dims, obj.xindexes.dims)
    return collapsible_section('Dimensions', inline_details=dim_list, enabled=False, collapsed=True)

def array_section(obj) -> str:
    if False:
        return 10
    data_id = 'section-' + str(uuid.uuid4())
    collapsed = 'checked' if _get_boolean_with_default('display_expand_data', default=True) else ''
    variable = getattr(obj, 'variable', obj)
    preview = escape(inline_variable_array_repr(variable, max_width=70))
    data_repr = short_data_repr_html(obj)
    data_icon = _icon('icon-database')
    return f"<div class='xr-array-wrap'><input id='{data_id}' class='xr-array-in' type='checkbox' {collapsed}><label for='{data_id}' title='Show/hide data repr'>{data_icon}</label><div class='xr-array-preview xr-preview'><span>{preview}</span></div><div class='xr-array-data'>{data_repr}</div></div>"
coord_section = partial(_mapping_section, name='Coordinates', details_func=summarize_coords, max_items_collapse=25, expand_option_name='display_expand_coords')
datavar_section = partial(_mapping_section, name='Data variables', details_func=summarize_vars, max_items_collapse=15, expand_option_name='display_expand_data_vars')
index_section = partial(_mapping_section, name='Indexes', details_func=summarize_indexes, max_items_collapse=0, expand_option_name='display_expand_indexes')
attr_section = partial(_mapping_section, name='Attributes', details_func=summarize_attrs, max_items_collapse=10, expand_option_name='display_expand_attrs')

def _get_indexes_dict(indexes):
    if False:
        i = 10
        return i + 15
    return {tuple(index_vars.keys()): idx for (idx, index_vars) in indexes.group_by_index()}

def _obj_repr(obj, header_components, sections):
    if False:
        for i in range(10):
            print('nop')
    'Return HTML repr of an xarray object.\n\n    If CSS is not injected (untrusted notebook), fallback to the plain text repr.\n\n    '
    header = f"<div class='xr-header'>{''.join((h for h in header_components))}</div>"
    sections = ''.join((f"<li class='xr-section-item'>{s}</li>" for s in sections))
    (icons_svg, css_style) = _load_static_files()
    return f"<div>{icons_svg}<style>{css_style}</style><pre class='xr-text-repr-fallback'>{escape(repr(obj))}</pre><div class='xr-wrap' style='display:none'>{header}<ul class='xr-sections'>{sections}</ul></div></div>"

def array_repr(arr) -> str:
    if False:
        for i in range(10):
            print('nop')
    dims = OrderedDict(((k, v) for (k, v) in zip(arr.dims, arr.shape)))
    if hasattr(arr, 'xindexes'):
        indexed_dims = arr.xindexes.dims
    else:
        indexed_dims = {}
    obj_type = f'xarray.{type(arr).__name__}'
    arr_name = f"'{arr.name}'" if getattr(arr, 'name', None) else ''
    header_components = [f"<div class='xr-obj-type'>{obj_type}</div>", f"<div class='xr-array-name'>{arr_name}</div>", format_dims(dims, indexed_dims)]
    sections = [array_section(arr)]
    if hasattr(arr, 'coords'):
        sections.append(coord_section(arr.coords))
    if hasattr(arr, 'xindexes'):
        indexes = _get_indexes_dict(arr.xindexes)
        sections.append(index_section(indexes))
    sections.append(attr_section(arr.attrs))
    return _obj_repr(arr, header_components, sections)

def dataset_repr(ds) -> str:
    if False:
        for i in range(10):
            print('nop')
    obj_type = f'xarray.{type(ds).__name__}'
    header_components = [f"<div class='xr-obj-type'>{escape(obj_type)}</div>"]
    sections = [dim_section(ds), coord_section(ds.coords), datavar_section(ds.data_vars), index_section(_get_indexes_dict(ds.xindexes)), attr_section(ds.attrs)]
    return _obj_repr(ds, header_components, sections)