from __future__ import annotations
import sys, os, inspect
from fastcore.utils import *
from fastcore.meta import delegates
__all__ = ['meta', 'div', 'img', 'btn', 'tbl_row', 'tbl_sep']

def meta(md, classes=None, style=None, **kwargs):
    if False:
        while True:
            i = 10
    'A metadata section for qmd div in `{}`'
    if style:
        kwargs['style'] = '; '.join((f'{k}: {v}' for (k, v) in style.items()))
    props = ' '.join((f'{k}="{v}"' for (k, v) in kwargs.items()))
    classes = ' '.join(('.' + c for c in L(classes)))
    meta = []
    if classes:
        meta.append(classes)
    if props:
        meta.append(props)
    meta = ' '.join(meta)
    return md + ('{' + meta + '}' if meta else '')

def div(txt, classes=None, style=None, **kwargs):
    if False:
        return 10
    'A qmd div with optional metadata section'
    return meta('::: ', classes=classes, style=style, **kwargs) + f'\n\n{txt}\n\n:::\n\n'

def img(fname, classes=None, style=None, height=None, relative=None, link=False, **kwargs):
    if False:
        return 10
    'A qmd image'
    (kwargs, style) = (kwargs or {}, style or {})
    if height:
        kwargs['height'] = f'{height}px'
    if relative:
        (pos, px) = relative
        style['position'] = 'relative'
        style[pos] = f'{px}px'
    res = meta(f'![]({fname})', classes=classes, style=style, **kwargs)
    return f'[{res}]({fname})' if link else res

def btn(txt, link, classes=None, style=None, **kwargs):
    if False:
        while True:
            i = 10
    'A qmd button'
    return meta(f'[{txt}]({link})', classes=classes, style=style, role='button')

def tbl_row(cols: list):
    if False:
        return 10
    'Create a markdown table row from `cols`'
    return '|' + '|'.join((str(c or '') for c in cols)) + '|'

def tbl_sep(sizes: int | list=3):
    if False:
        print('Hello World!')
    'Create a markdown table separator with relative column size `sizes`'
    if isinstance(sizes, int):
        sizes = [3] * sizes
    return tbl_row(('-' * s for s in sizes))

def _install_nbdev():
    if False:
        i = 10
        return i + 15
    return div('#### pip\n\n```sh\npip install -U nbdev\n```\n\n#### conda\n\n```sh\nconda install -c fastai nbdev\n```\n', ['panel-tabset'])