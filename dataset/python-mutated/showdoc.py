from __future__ import annotations
from .doclinks import *
from .config import get_config
from fastcore.dispatch import TypeDispatch
from fastcore.docments import *
from fastcore.utils import *
from importlib import import_module
import inspect, sys
from collections import OrderedDict
from textwrap import fill
from types import FunctionType
__all__ = ['DocmentTbl', 'ShowDocRenderer', 'BasicMarkdownRenderer', 'show_doc', 'BasicHtmlRenderer', 'doc', 'showdoc_nm', 'colab_link']

def _non_empty_keys(d: dict):
    if False:
        while True:
            i = 10
    return L([k for (k, v) in d.items() if v != inspect._empty])

def _bold(s):
    if False:
        i = 10
        return i + 15
    return f'**{s}**' if s.strip() else s

def _escape_markdown(s):
    if False:
        return 10
    for c in '|^':
        s = re.sub(f'\\\\?\\{c}', f'\\{c}', s)
    return s.replace('\n', '<br>')

def _maybe_nm(o):
    if False:
        for i in range(10):
            print('nop')
    if o == inspect._empty:
        return ''
    else:
        return o.__name__ if hasattr(o, '__name__') else _escape_markdown(str(o))

def _list2row(l: list):
    if False:
        i = 10
        return i + 15
    return '| ' + ' | '.join([_maybe_nm(o) for o in l]) + ' |'

class DocmentTbl:
    _map = OrderedDict({'anno': 'Type', 'default': 'Default', 'docment': 'Details'})

    def __init__(self, obj, verbose=True, returns=True):
        if False:
            for i in range(10):
                print('nop')
        'Compute the docment table string'
        self.verbose = verbose
        self.returns = False if isdataclass(obj) else returns
        try:
            self.params = L(signature_ex(obj, eval_str=True).parameters.keys())
        except (ValueError, TypeError):
            self.params = []
        try:
            _dm = docments(obj, full=True, returns=returns)
        except:
            _dm = {}
        if 'self' in _dm:
            del _dm['self']
        for d in _dm.values():
            d['docment'] = ifnone(d['docment'], inspect._empty)
        self.dm = _dm

    @property
    def _columns(self):
        if False:
            for i in range(10):
                print('nop')
        "Compute the set of fields that have at least one non-empty value so we don't show tables empty columns"
        cols = set(flatten(L(self.dm.values()).filter().map(_non_empty_keys)))
        candidates = self._map if self.verbose else {'docment': 'Details'}
        return OrderedDict({k: v for (k, v) in candidates.items() if k in cols})

    @property
    def has_docment(self):
        if False:
            for i in range(10):
                print('nop')
        return 'docment' in self._columns and self._row_list

    @property
    def has_return(self):
        if False:
            print('Hello World!')
        return self.returns and bool(_non_empty_keys(self.dm.get('return', {})))

    def _row(self, nm, props):
        if False:
            return 10
        'unpack data for single row to correspond with column names.'
        return [nm] + [props[c] for c in self._columns]

    @property
    def _row_list(self):
        if False:
            for i in range(10):
                print('nop')
        'unpack data for all rows.'
        ordered_params = [(p, self.dm[p]) for p in self.params if p != 'self' and p in self.dm]
        return L([self._row(nm, props) for (nm, props) in ordered_params])

    @property
    def _hdr_list(self):
        if False:
            print('Hello World!')
        return ['  '] + [_bold(l) for l in L(self._columns.values())]

    @property
    def hdr_str(self):
        if False:
            return 10
        'The markdown string for the header portion of the table'
        md = _list2row(self._hdr_list)
        return md + '\n' + _list2row(['-' * len(l) for l in self._hdr_list])

    @property
    def params_str(self):
        if False:
            while True:
                i = 10
        'The markdown string for the parameters portion of the table.'
        return '\n'.join(self._row_list.map(_list2row))

    @property
    def return_str(self):
        if False:
            while True:
                i = 10
        'The markdown string for the returns portion of the table.'
        return _list2row(['**Returns**'] + [_bold(_maybe_nm(self.dm['return'][c])) for c in self._columns])

    def _repr_markdown_(self):
        if False:
            while True:
                i = 10
        if not self.has_docment:
            return ''
        _tbl = [self.hdr_str, self.params_str]
        if self.has_return:
            _tbl.append(self.return_str)
        return '\n'.join(_tbl)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.__str__() == str(other).strip()
    __str__ = _repr_markdown_
    __repr__ = basic_repr()

def _docstring(sym):
    if False:
        print('Hello World!')
    npdoc = parse_docstring(sym)
    return '\n\n'.join([npdoc['Summary'], npdoc['Extended']]).strip()

def _fullname(o):
    if False:
        print('Hello World!')
    (module, name) = (getattr(o, '__module__', None), qual_name(o))
    return name if module is None or module in ('__main__', 'builtins') else module + '.' + name

class ShowDocRenderer:

    def __init__(self, sym, name: str | None=None, title_level: int=3):
        if False:
            while True:
                i = 10
        'Show documentation for `sym`'
        sym = getattr(sym, '__wrapped__', sym)
        sym = getattr(sym, 'fget', None) or getattr(sym, 'fset', None) or sym
        store_attr()
        self.nm = name or qual_name(sym)
        self.isfunc = inspect.isfunction(sym)
        try:
            self.sig = signature_ex(sym, eval_str=True)
        except (ValueError, TypeError):
            self.sig = None
        self.docs = _docstring(sym)
        self.dm = DocmentTbl(sym)
        self.fn = _fullname(sym)
    __repr__ = basic_repr()

def _f_name(o):
    if False:
        i = 10
        return i + 15
    return f'<function {o.__name__}>' if isinstance(o, FunctionType) else None

def _fmt_anno(o):
    if False:
        i = 10
        return i + 15
    return inspect.formatannotation(o).strip("'").replace(' ', '')

def _show_param(param):
    if False:
        return 10
    'Like `Parameter.__str__` except removes: quotes in annos, spaces, ids in reprs'
    (kind, res, anno, default) = (param.kind, param._name, param._annotation, param._default)
    kind = '*' if kind == inspect._VAR_POSITIONAL else '**' if kind == inspect._VAR_KEYWORD else ''
    res = kind + res
    if anno is not inspect._empty:
        res += f':{_f_name(anno) or _fmt_anno(anno)}'
    if default is not inspect._empty:
        res += f'={_f_name(default) or repr(default)}'
    return res

def _fmt_sig(sig):
    if False:
        return 10
    if sig is None:
        return ''
    p = {k: v for (k, v) in sig.parameters.items()}
    _params = [_show_param(p[k]) for k in p.keys() if k != 'self']
    return '(' + ', '.join(_params) + ')'

def _wrap_sig(s):
    if False:
        print('Hello World!')
    'wrap a signature to appear on multiple lines if necessary.'
    pad = '> ' + ' ' * 5
    indent = pad + ' ' * (s.find('(') + 1)
    return fill(s, width=80, initial_indent=pad, subsequent_indent=indent)

def _ext_link(url, txt, xtra=''):
    if False:
        print('Hello World!')
    return f'[{txt}]({url}){{target="_blank" {xtra}}}'

class BasicMarkdownRenderer(ShowDocRenderer):
    """Markdown renderer for `show_doc`"""

    def _repr_markdown_(self):
        if False:
            for i in range(10):
                print('nop')
        doc = '---\n\n'
        src = NbdevLookup().code(self.fn)
        if src:
            doc += _ext_link(src, 'source', 'style="float:right; font-size:smaller"') + '\n\n'
        h = '#' * self.title_level
        doc += f'{h} {self.nm}\n\n'
        sig = _wrap_sig(f'{self.nm} {_fmt_sig(self.sig)}') if self.sig else ''
        doc += f'{sig}'
        if self.docs:
            doc += f'\n\n{self.docs}'
        if self.dm.has_docment:
            doc += f'\n\n{self.dm}'
        return doc
    __repr__ = __str__ = _repr_markdown_

def show_doc(sym, renderer=None, name: str | None=None, title_level: int=3):
    if False:
        for i in range(10):
            print('nop')
    'Show signature and docstring for `sym`'
    if renderer is None:
        renderer = get_config().get('renderer', None)
    if renderer is None:
        renderer = BasicMarkdownRenderer
    elif isinstance(renderer, str):
        (p, m) = renderer.rsplit('.', 1)
        renderer = getattr(import_module(p), m)
    if isinstance(sym, TypeDispatch):
        pass
    else:
        return renderer(sym or show_doc, name=name, title_level=title_level)

def _html_link(url, txt):
    if False:
        return 10
    return f'<a href="{url}" target="_blank" rel="noreferrer noopener">{txt}</a>'

class BasicHtmlRenderer(ShowDocRenderer):
    """Simple HTML renderer for `show_doc`"""

    def _repr_html_(self):
        if False:
            print('Hello World!')
        doc = '<hr/>\n'
        doc += f'<h{self.title_level}>{self.nm}</h{self.title_level}>\n'
        doc += f'<blockquote><pre><code>{self.nm}{_fmt_sig(self.sig)}</code></pre></blockquote>'
        if self.docs:
            doc += f'<p>{self.docs}</p>'
        return doc

    def doc(self):
        if False:
            return 10
        'Show `show_doc` info along with link to docs'
        from IPython.display import display, HTML
        res = self._repr_html_()
        docs = NbdevLookup().doc(self.fn)
        if docs is not None:
            res += '\n<p>' + _html_link(docs, 'Show in docs') + '</p>'
        display(HTML(res))

def doc(elt):
    if False:
        i = 10
        return i + 15
    'Show `show_doc` info along with link to docs'
    BasicHtmlRenderer(elt).doc()

def showdoc_nm(tree):
    if False:
        print('Hello World!')
    'Get the fully qualified name for showdoc.'
    return ifnone(patch_name(tree), tree.name)

def colab_link(path):
    if False:
        print('Hello World!')
    'Get a link to the notebook at `path` on Colab'
    from IPython.display import Markdown
    cfg = get_config()
    pre = 'https://colab.research.google.com/github/'
    res = f'{pre}{cfg.user}/{cfg.lib_name}/blob/{cfg.branch}/{cfg.nbs_path.name}/{path}.ipynb'
    display(Markdown(f'[Open `{path}` in Colab]({res})'))