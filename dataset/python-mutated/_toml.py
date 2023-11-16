from contextlib import contextmanager
from funcy import reraise
from ._common import ParseError, _dump_data, _load_data, _modify_data

class TOMLFileCorruptedError(ParseError):

    def __init__(self, path):
        if False:
            while True:
                i = 10
        super().__init__(path, 'TOML file structure is corrupted')

def load_toml(path, fs=None, **kwargs):
    if False:
        i = 10
        return i + 15
    return _load_data(path, parser=parse_toml, fs=fs, **kwargs)

def _parse_toml(text, path):
    if False:
        print('Hello World!')
    from tomlkit import loads
    from tomlkit.exceptions import ParseError as TomlkitParseError
    with reraise(TomlkitParseError, TOMLFileCorruptedError(path)):
        return loads(text)

def parse_toml(text, path, preserve_comments=False):
    if False:
        i = 10
        return i + 15
    rval = _parse_toml(text, path)
    if preserve_comments:
        return rval
    return rval.unwrap()

def parse_toml_for_update(text, path):
    if False:
        return 10
    return parse_toml(text, path, preserve_comments=True)

def _dump(data, stream, sort_keys=False):
    if False:
        return 10
    import tomlkit
    return tomlkit.dump(data, stream, sort_keys=sort_keys)

def dump_toml(path, data, fs=None, **kwargs):
    if False:
        return 10
    return _dump_data(path, data, dumper=_dump, fs=fs, **kwargs)

@contextmanager
def modify_toml(path, fs=None):
    if False:
        i = 10
        return i + 15
    with _modify_data(path, parse_toml_for_update, _dump, fs=fs) as d:
        yield d