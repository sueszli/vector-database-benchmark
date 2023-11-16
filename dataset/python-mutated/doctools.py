from __future__ import annotations
import re
import typing
from functools import lru_cache
from textwrap import dedent, indent, wrap
import numpy as np
if typing.TYPE_CHECKING:
    from typing import Any, Type, TypeVar
    from plotnine.typing import Geom, Scale, Stat
    T = TypeVar('T')
common_geom_params = ['mapping', 'data', 'stat', 'position', 'na_rm', 'inherit_aes', 'show_legend', 'raster']
common_geom_param_values = {'mapping': None, 'data': None, 'inherit_aes': True, 'show_legend': None, 'raster': False}
common_stat_params = ['mapping', 'data', 'geom', 'position', 'na_rm']
common_stat_param_values = common_geom_param_values
GEOM_SIGNATURE_TPL = '\n.. rubric:: Usage\n\n::\n\n    {signature}\n\nOnly the ``data`` and ``mapping`` can be positional, the rest must\nbe keyword arguments. ``**kwargs`` can be aesthetics (or parameters)\nused by the ``stat``.\n'
AESTHETICS_TABLE_TPL = '\n{table}\n\nThe **bold** aesthetics are required.\n'
STAT_SIGNATURE_TPL = '\n.. rubric:: Usage\n\n::\n\n    {signature}\n\nOnly the ``mapping`` and ``data`` can be positional, the rest must\nbe keyword arguments. ``**kwargs`` can be aesthetics (or parameters)\nused by the ``geom``.\n'
common_params_doc = {'mapping': 'Aesthetic mappings created with :meth:`~plotnine.aes`. If specified and :py:`inherit.aes=True`, it is combined with the default mapping for the plot. You must supply mapping if there is no plot mapping.', 'data': 'The data to be displayed in this layer. If :py:`None`, the data from from the :py:`ggplot()` call is used. If specified, it overrides the data from the :py:`ggplot()` call.', 'stat': 'The statistical transformation to use on the data for this layer. If it is a string, it must be the registered and known to Plotnine.', 'position': 'Position adjustment. If it is a string, it must be registered and known to Plotnine.', 'na_rm': 'If :py:`False`, removes missing values with a warning. If :py:`True` silently removes missing values.', 'inherit_aes': 'If :py:`False`, overrides the default aesthetics.', 'show_legend': "Whether this layer should be included in the legends. :py:`None` the default, includes any aesthetics that are mapped. If a :class:`bool`, :py:`False` never includes and :py:`True` always includes. A :class:`dict` can be used to *exclude* specific aesthetis of the layer from showing in the legend. e.g :py:`show_legend={'color': False}`, any other aesthetic are included by default.", 'raster': 'If ``True``, draw onto this layer a raster (bitmap) object even ifthe final image is in vector format.'}
GEOM_PARAMS_TPL = 'mapping : aes, optional\n    {mapping}\n    {_aesthetics_doc}\ndata : dataframe, optional\n    {data}\nstat : str or stat, optional (default: {default_stat})\n    {stat}\nposition : str or position, optional (default: {default_position})\n    {position}\nna_rm : bool, optional (default: {default_na_rm})\n    {na_rm}\ninherit_aes : bool, optional (default: {default_inherit_aes})\n    {inherit_aes}\nshow_legend : bool or dict, optional (default: None)\n    {show_legend}\nraster : bool, optional (default: {default_raster})\n    {raster}\n'
STAT_PARAMS_TPL = 'mapping : aes, optional\n    {mapping}\n    {_aesthetics_doc}\ndata : dataframe, optional\n    {data}\ngeom : str or geom, optional (default: {default_geom})\n    {stat}\nposition : str or position, optional (default: {default_position})\n    {position}\nna_rm : bool, optional (default: {default_na_rm})\n    {na_rm}\n'
DOCSTRING_SECTIONS = {'parameters', 'see also', 'note', 'notes', 'example', 'examples'}
PARAM_PATTERN = re.compile('\\s*([_A-Za-z]\\w*)\\s:\\s')

def dict_to_table(header: tuple[str, str], contents: dict[str, str]) -> str:
    if False:
        return 10
    '\n    Convert dict to an (n x 2) table\n\n    Parameters\n    ----------\n    header : tuple\n        Table header. Should have a length of 2.\n    contents : dict\n        The key becomes column 1 of table and the\n        value becomes column 2 of table.\n\n    Examples\n    --------\n    >>> d = {"alpha": 1, "color": "blue", "fill": None}\n    >>> print(dict_to_table(("Aesthetic", "Default Value"), d))\n    ========= =========\n    Aesthetic Default Value\n    ========= =========\n    alpha     :py:`1`\n    color     :py:`\'blue\'`\n    fill      :py:`None`\n    ========= =========\n    '

    def to_text(row: tuple[str, str]) -> str:
        if False:
            i = 10
            return i + 15
        (name, value) = row
        m = max_col1_size + 1 - len(name)
        spacing = ' ' * m
        return ''.join([name, spacing, value])

    def longest_value(row: tuple[str, str]) -> int:
        if False:
            for i in range(10):
                print('nop')
        return max((len(value) for value in row))
    rows = []
    for (name, value) in contents.items():
        if value != '':
            if isinstance(value, str):
                value = f"'{value}'"
            value = f':py:`{value}`'
        rows.append((name, value))
    n = max((longest_value(row) for row in [header] + rows))
    hborder = ('=' * n, '=' * n)
    rows = [hborder, header, hborder] + rows + [hborder]
    max_col1_size = np.max([len(col1) for (col1, _) in rows])
    table = '\n'.join([to_text(row) for row in rows])
    return table

def make_signature(name: str, params: dict[str, Any], common_params: list[str], common_param_values: dict[str, Any]) -> str:
    if False:
        print('Hello World!')
    '\n    Create a signature for a geom or stat\n\n    Gets the DEFAULT_PARAMS (params) and creates are comma\n    separated list of the `name=value` pairs. The common_params\n    come first in the list, and they get take their values from\n    either the params-dict or the common_geom_param_values-dict.\n    '
    tokens = []
    seen = set()

    def tokens_append(key: str, value: Any):
        if False:
            while True:
                i = 10
        if isinstance(value, str):
            value = f"'{value}'"
        tokens.append(f'{key}={value}')
    for key in common_params:
        seen.add(key)
        try:
            value = params[key]
        except KeyError:
            value = common_param_values[key]
        tokens_append(key, value)
    for key in set(params) - seen:
        tokens_append(key, params[key])
    s_params = ', '.join(tokens)
    s1 = f'{name}('
    s2 = f'{s_params}, **kwargs)'
    line_width = 78 - len(s1)
    indent_spaces = ' ' * (len(s1) + 4)
    s2_lines = wrap(s2, width=line_width)
    s2_indented = f'\n{indent_spaces}'.join(s2_lines)
    return f'{s1}{s2_indented}'

@lru_cache(maxsize=256)
def docstring_section_lines(docstring: str, section_name: str) -> str:
    if False:
        print('Hello World!')
    '\n    Return a section of a numpydoc string\n\n    Paramters\n    ---------\n    docstring : str\n        Docstring\n    section_name : str\n        Name of section to return\n\n    Returns\n    -------\n    section : str\n        Section minus the header\n    '
    lines = []
    inside_section = False
    underline = '-' * len(section_name)
    expect_underline = False
    for line in docstring.splitlines():
        _line = line.strip().lower()
        if expect_underline:
            expect_underline = False
            if _line == underline:
                inside_section = True
                continue
        if _line == section_name:
            expect_underline = True
        elif _line in DOCSTRING_SECTIONS:
            break
        elif inside_section:
            lines.append(line)
    return '\n'.join(lines)

def docstring_parameters_section(obj: Any) -> str:
    if False:
        while True:
            i = 10
    '\n    Return the parameters section of a docstring\n    '
    return docstring_section_lines(obj.__doc__, 'parameters')

def param_spec(line: str) -> str | None:
    if False:
        return 10
    '\n    Identify and return parameter\n\n    Parameters\n    ----------\n    line : str\n        A line in the parameter section.\n\n    Returns\n    -------\n    name : str or None\n        Name of the parameter if the line for the parameter\n        type specification and None otherwise.\n\n    Examples\n    --------\n    >>> param_spec(\'line : str\')\n    breaks\n    >>> param_spec("    A line in the parameter section.")\n    '
    m = PARAM_PATTERN.match(line)
    return m.group(1) if m else None

def parameters_str_to_dict(param_section: str) -> dict[str, str]:
    if False:
        return 10
    '\n    Convert a param section to a dict\n\n    Parameters\n    ----------\n    param_section : str\n        Text in the parameter section\n\n    Returns\n    -------\n    d : dict\n        Dictionary of the parameters in the order that they\n        are described in the parameters section. The dict\n        is of the form ``{param: all_parameter_text}``.\n        You can reconstruct the ``param_section`` from the\n        keys of the dictionary.\n\n    See Also\n    --------\n    :func:`parameters_dict_to_str`\n    '
    d = {}
    previous_param = ''
    param_desc: list[str] = []
    for line in param_section.split('\n'):
        param = param_spec(line)
        if param:
            if previous_param:
                d[previous_param] = '\n'.join(param_desc)
            param_desc = [line]
            previous_param = param
        elif param_desc:
            param_desc.append(line)
    if previous_param:
        d[previous_param] = '\n'.join(param_desc)
    return d

def parameters_dict_to_str(d: dict[str, str]) -> str:
    if False:
        while True:
            i = 10
    '\n    Convert a dict of param section to a string\n\n    Parameters\n    ----------\n    d : dict\n        Parameters and their descriptions in a docstring\n\n    Returns\n    -------\n    param_section : str\n        Text in the parameter section\n\n    See Also\n    --------\n    :func:`parameters_str_to_dict`\n    '
    return '\n'.join(d.values())

def qualified_name(s: str | type | object, prefix: str) -> str:
    if False:
        while True:
            i = 10
    "\n    Return the qualified name of s\n\n    Only if s does not start with the prefix\n\n    Examples\n    --------\n    >>> qualified_name('bin', 'stat_')\n    '~plotnine.stats.stat_bin'\n    >>> qualified_name('point', 'geom_')\n    '~plotnine.geoms.geom_point'\n    >>> qualified_name('stack', 'position_')\n    '~plotnine.positions.position_'\n    "
    lookup = {'stat_': '~plotnine.stats.stat_', 'geom_': '~plotnine.geoms.geom_', 'position_': '~plotnine.positions.position_'}
    if isinstance(s, str):
        if not s.startswith(prefix) and prefix in lookup:
            pre = lookup[prefix]
            s = f'{pre}{s}'
    elif isinstance(s, type):
        s = s.__name__
    else:
        s = s.__class__.__name__
    return s

def document_geom(geom: type[Geom]) -> type[Geom]:
    if False:
        while True:
            i = 10
    '\n    Create a structured documentation for the geom\n\n    It replaces `{usage}`, `{common_parameters}` and\n    `{aesthetics}` with generated documentation.\n    '
    docstring = dedent(geom.__doc__ or '')
    signature = make_signature(geom.__name__, geom.DEFAULT_PARAMS, common_geom_params, common_geom_param_values)
    usage = GEOM_SIGNATURE_TPL.format(signature=signature)
    contents = {f'**{ae}**': '' for ae in sorted(geom.REQUIRED_AES)}
    if geom.DEFAULT_AES:
        d = geom.DEFAULT_AES.copy()
        d['group'] = ''
        contents.update(sorted(d.items()))
    table = dict_to_table(('Aesthetic', 'Default value'), contents)
    aesthetics_table = AESTHETICS_TABLE_TPL.format(table=table)
    tpl = dedent(geom._aesthetics_doc.lstrip('\n'))
    aesthetics_doc = tpl.format(aesthetics_table=aesthetics_table)
    aesthetics_doc = indent(aesthetics_doc, ' ' * 4)
    d = geom.DEFAULT_PARAMS
    common_parameters = GEOM_PARAMS_TPL.format(default_stat=qualified_name(d['stat'], 'stat_'), default_position=qualified_name(d['position'], 'position_'), default_na_rm=d['na_rm'], default_inherit_aes=d.get('inherit_aes', True), default_raster=d.get('raster', False), _aesthetics_doc=aesthetics_doc, **common_params_doc)
    docstring = docstring.replace('{usage}', usage)
    docstring = docstring.replace('{common_parameters}', common_parameters)
    geom.__doc__ = docstring
    return geom

def document_stat(stat: type[Stat]) -> type[Stat]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a structured documentation for the stat\n\n    It replaces `{usage}`, `{common_parameters}` and\n    `{aesthetics}` with generated documentation.\n    '
    docstring = dedent(stat.__doc__ or '')
    signature = make_signature(stat.__name__, stat.DEFAULT_PARAMS, common_stat_params, common_stat_param_values)
    usage = STAT_SIGNATURE_TPL.format(signature=signature)
    contents = {f'**{ae}**': '' for ae in sorted(stat.REQUIRED_AES)}
    contents.update(sorted(stat.DEFAULT_AES.items()))
    table = dict_to_table(('Aesthetic', 'Default value'), contents)
    aesthetics_table = AESTHETICS_TABLE_TPL.format(table=table)
    tpl = dedent(stat._aesthetics_doc.lstrip('\n'))
    aesthetics_doc = tpl.format(aesthetics_table=aesthetics_table)
    aesthetics_doc = indent(aesthetics_doc, ' ' * 4)
    d = stat.DEFAULT_PARAMS
    common_parameters = STAT_PARAMS_TPL.format(default_geom=qualified_name(d['geom'], 'geom_'), default_position=qualified_name(d['position'], 'position_'), default_na_rm=d['na_rm'], _aesthetics_doc=aesthetics_doc, **common_params_doc)
    docstring = docstring.replace('{usage}', usage)
    docstring = docstring.replace('{common_parameters}', common_parameters)
    stat.__doc__ = docstring
    return stat

def document_scale(cls: type[Scale]) -> type[Scale]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a documentation for a scale\n\n    Import the superclass parameters\n\n    It replaces `{superclass_parameters}` with the documentation\n    of the parameters from the superclass.\n\n    Parameters\n    ----------\n    cls : type\n        A scale class\n\n    Returns\n    -------\n    cls : type\n        The scale class with a modified docstring.\n    '
    params_list = []
    cls_param_string = docstring_parameters_section(cls)
    cls_param_dict = parameters_str_to_dict(cls_param_string)
    cls_params = set(cls_param_dict.keys())
    for (i, base) in enumerate(cls.__bases__):
        base_param_string = param_string = docstring_parameters_section(base)
        base_param_dict = parameters_str_to_dict(base_param_string)
        base_params = set(base_param_dict.keys())
        duplicate_params = base_params & cls_params
        for param in duplicate_params:
            del base_param_dict[param]
        if duplicate_params:
            param_string = parameters_dict_to_str(base_param_dict)
        if i == 0:
            param_string = param_string.strip()
        params_list.append(param_string)
        cls_params |= base_params
    superclass_parameters = '\n'.join(params_list)
    cls_doc = cls.__doc__ or ''
    cls.__doc__ = cls_doc.format(superclass_parameters=superclass_parameters)
    return cls
DOC_FUNCTIONS = {'geom': document_geom, 'stat': document_stat, 'scale': document_scale}

def document(cls: Type[T]) -> Type[T]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Document a plotnine class\n\n    To be used as a decorator\n    '
    if cls.__doc__ is None:
        return cls
    baseclass_name = cls.mro()[-2].__name__
    try:
        return DOC_FUNCTIONS[baseclass_name](cls)
    except KeyError:
        return cls