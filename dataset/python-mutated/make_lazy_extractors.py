import os
import shutil
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inspect import getsource
from devscripts.utils import get_filename_args, read_file, write_file
NO_ATTR = object()
STATIC_CLASS_PROPERTIES = ['IE_NAME', '_ENABLED', '_VALID_URL', '_WORKING', 'IE_DESC', '_NETRC_MACHINE', 'SEARCH_KEY', 'age_limit', '_RETURN_TYPE']
CLASS_METHODS = ['ie_key', 'suitable', '_match_valid_url', 'working', 'get_temp_id', '_match_id', 'description', 'is_suitable', 'supports_login', 'is_single_video']
IE_TEMPLATE = '\nclass {name}({bases}):\n    _module = {module!r}\n'
MODULE_TEMPLATE = read_file('devscripts/lazy_load_template.py')

def main():
    if False:
        while True:
            i = 10
    lazy_extractors_filename = get_filename_args(default_outfile='yt_dlp/extractor/lazy_extractors.py')
    if os.path.exists(lazy_extractors_filename):
        os.remove(lazy_extractors_filename)
    _ALL_CLASSES = get_all_ies()
    import yt_dlp.plugins
    from yt_dlp.extractor.common import InfoExtractor, SearchInfoExtractor
    _ALL_CLASSES = [cls for cls in _ALL_CLASSES if not cls.__module__.startswith(f'{yt_dlp.plugins.PACKAGE_NAME}.')]
    DummyInfoExtractor = type('InfoExtractor', (InfoExtractor,), {'IE_NAME': NO_ATTR})
    module_src = '\n'.join((MODULE_TEMPLATE, '    _module = None', *extra_ie_code(DummyInfoExtractor), '\nclass LazyLoadSearchExtractor(LazyLoadExtractor):\n    pass\n', *build_ies(_ALL_CLASSES, (InfoExtractor, SearchInfoExtractor), DummyInfoExtractor)))
    write_file(lazy_extractors_filename, f'{module_src}\n')

def get_all_ies():
    if False:
        i = 10
        return i + 15
    PLUGINS_DIRNAME = 'ytdlp_plugins'
    BLOCKED_DIRNAME = f'{PLUGINS_DIRNAME}_blocked'
    if os.path.exists(PLUGINS_DIRNAME):
        shutil.move(PLUGINS_DIRNAME, BLOCKED_DIRNAME)
    try:
        from yt_dlp.extractor.extractors import _ALL_CLASSES
    finally:
        if os.path.exists(BLOCKED_DIRNAME):
            shutil.move(BLOCKED_DIRNAME, PLUGINS_DIRNAME)
    return _ALL_CLASSES

def extra_ie_code(ie, base=None):
    if False:
        return 10
    for var in STATIC_CLASS_PROPERTIES:
        val = getattr(ie, var)
        if val != (getattr(base, var) if base else NO_ATTR):
            yield f'    {var} = {val!r}'
    yield ''
    for name in CLASS_METHODS:
        f = getattr(ie, name)
        if not base or f.__func__ != getattr(base, name).__func__:
            yield getsource(f)

def build_ies(ies, bases, attr_base):
    if False:
        i = 10
        return i + 15
    names = []
    for ie in sort_ies(ies, bases):
        yield build_lazy_ie(ie, ie.__name__, attr_base)
        if ie in ies:
            names.append(ie.__name__)
    yield f"\n_ALL_CLASSES = [{', '.join(names)}]"

def sort_ies(ies, ignored_bases):
    if False:
        print('Hello World!')
    'find the correct sorting and add the required base classes so that subclasses can be correctly created'
    (classes, returned_classes) = (ies[:-1], set())
    assert ies[-1].__name__ == 'GenericIE', 'Last IE must be GenericIE'
    while classes:
        for c in classes[:]:
            bases = set(c.__bases__) - {object, *ignored_bases}
            restart = False
            for b in sorted(bases, key=lambda x: x.__name__):
                if b not in classes and b not in returned_classes:
                    assert b.__name__ != 'GenericIE', 'Cannot inherit from GenericIE'
                    classes.insert(0, b)
                    restart = True
            if restart:
                break
            if bases <= returned_classes:
                yield c
                returned_classes.add(c)
                classes.remove(c)
                break
    yield ies[-1]

def build_lazy_ie(ie, name, attr_base):
    if False:
        for i in range(10):
            print('nop')
    bases = ', '.join(({'InfoExtractor': 'LazyLoadExtractor', 'SearchInfoExtractor': 'LazyLoadSearchExtractor'}.get(base.__name__, base.__name__) for base in ie.__bases__))
    s = IE_TEMPLATE.format(name=name, module=ie.__module__, bases=bases)
    return s + '\n'.join(extra_ie_code(ie, attr_base))
if __name__ == '__main__':
    main()