"""functions to update rc files"""
import os
import re
import typing as tp
RENDERERS: list[tp.Callable] = []

def renderer(f):
    if False:
        print('Hello World!')
    'Adds decorated function to renderers list.'
    RENDERERS.append(f)

@renderer
def prompt(config):
    if False:
        i = 10
        return i + 15
    prompt = config.get('prompt')
    if prompt:
        yield f'$PROMPT = {prompt!r}'

@renderer
def colors(config):
    if False:
        print('Hello World!')
    style = config.get('color')
    if style:
        yield f'$XONSH_COLOR_STYLE = {style!r}'

@renderer
def xontribs(config):
    if False:
        i = 10
        return i + 15
    xtribs = config.get('xontribs')
    if xtribs:
        yield ('xontrib load ' + ' '.join(xtribs))

def config_to_xonsh(config, prefix='# XONSH WEBCONFIG START', suffix='# XONSH WEBCONFIG END'):
    if False:
        while True:
            i = 10
    'Turns config dict into xonsh code (str).'
    lines = [prefix]
    for func in RENDERERS:
        lines.extend(func(config))
    lines.append(suffix)
    return re.sub('\\\\r', '', '\n'.join(lines))

def insert_into_xonshrc(config, xonshrc='~/.xonshrc', prefix='# XONSH WEBCONFIG START', suffix='# XONSH WEBCONFIG END'):
    if False:
        return 10
    'Places a config dict into the xonshrc.'
    fname = os.path.expanduser(xonshrc)
    if os.path.isfile(fname):
        with open(fname) as f:
            s = f.read()
        (before, _, s) = s.partition(prefix)
        (_, _, after) = s.partition(suffix)
    else:
        before = after = ''
        dname = os.path.dirname(fname)
        if dname:
            os.makedirs(dname, exist_ok=True)
    new = config_to_xonsh(config, prefix=prefix, suffix=suffix)
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(before + new + after)
    return fname