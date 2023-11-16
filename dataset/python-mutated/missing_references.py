"""
This is a sphinx extension to freeze your broken reference problems
when using ``nitpicky = True``.

The basic operation is:

1. Add this extension to your ``conf.py`` extensions.
2. Add ``missing_references_write_json = True`` to your ``conf.py``
3. Run sphinx-build. It will generate ``missing-references.json``
    next to your ``conf.py``.
4. Remove ``missing_references_write_json = True`` from your
    ``conf.py`` (or set it to ``False``)
5. Run sphinx-build again, and ``nitpick_ignore`` will
    contain all of the previously failed references.

"""
from collections import defaultdict
import json
import logging
from pathlib import Path
from docutils.utils import get_source_line
from docutils import nodes
from sphinx.util import logging as sphinx_logging
import matplotlib
logger = sphinx_logging.getLogger(__name__)

class MissingReferenceFilter(logging.Filter):
    """
    A logging filter designed to record missing reference warning messages
    for use by this extension
    """

    def __init__(self, app):
        if False:
            print('Hello World!')
        self.app = app
        super().__init__()

    def _record_reference(self, record):
        if False:
            while True:
                i = 10
        if not (getattr(record, 'type', '') == 'ref' and isinstance(getattr(record, 'location', None), nodes.Node)):
            return
        if not hasattr(self.app.env, 'missing_references_warnings'):
            self.app.env.missing_references_warnings = defaultdict(set)
        record_missing_reference(self.app, self.app.env.missing_references_warnings, record.location)

    def filter(self, record):
        if False:
            for i in range(10):
                print('nop')
        self._record_reference(record)
        return True

def record_missing_reference(app, record, node):
    if False:
        print('Hello World!')
    domain = node['refdomain']
    typ = node['reftype']
    target = node['reftarget']
    location = get_location(node, app)
    domain_type = f'{domain}:{typ}'
    record[domain_type, target].add(location)

def record_missing_reference_handler(app, env, node, contnode):
    if False:
        i = 10
        return i + 15
    '\n    When the sphinx app notices a missing reference, it emits an\n    event which calls this function. This function records the missing\n    references for analysis at the end of the sphinx build.\n    '
    if not app.config.missing_references_enabled:
        return
    if not hasattr(env, 'missing_references_events'):
        env.missing_references_events = defaultdict(set)
    record_missing_reference(app, env.missing_references_events, node)

def get_location(node, app):
    if False:
        i = 10
        return i + 15
    '\n    Given a docutils node and a sphinx application, return a string\n    representation of the source location of this node.\n\n    Usually, this will be of the form "path/to/file:linenumber". Two\n    special values can be emitted, "<external>" for paths which are\n    not contained in this source tree (e.g. docstrings included from\n    other modules) or "<unknown>", indicating that the sphinx application\n    cannot locate the original source file (usually because an extension\n    has injected text into the sphinx parsing engine).\n    '
    (source, line) = get_source_line(node)
    if source:
        if ':docstring of' in source:
            (path, *post) = source.rpartition(':docstring of')
            post = ''.join(post)
        else:
            path = source
            post = ''
        basepath = Path(app.srcdir).parent.resolve()
        fullpath = Path(path).resolve()
        try:
            path = fullpath.relative_to(basepath)
        except ValueError:
            path = Path('<external>') / fullpath.name
        path = path.as_posix()
    else:
        path = '<unknown>'
        post = ''
    if not line:
        line = ''
    return f'{path}{post}:{line}'

def _truncate_location(location):
    if False:
        while True:
            i = 10
    '\n    Cuts off anything after the first colon in location strings.\n\n    This allows for easy comparison even when line numbers change\n    (as they do regularly).\n    '
    return location.split(':', 1)[0]

def _warn_unused_missing_references(app):
    if False:
        for i in range(10):
            print('nop')
    if not app.config.missing_references_warn_unused_ignores:
        return
    basepath = Path(matplotlib.__file__).parent.parent.parent.resolve()
    srcpath = Path(app.srcdir).parent.resolve()
    if basepath != srcpath:
        return
    references_ignored = getattr(app.env, 'missing_references_ignored_references', {})
    references_events = getattr(app.env, 'missing_references_events', {})
    for ((domain_type, target), locations) in references_ignored.items():
        missing_reference_locations = [_truncate_location(location) for location in references_events.get((domain_type, target), [])]
        for ignored_reference_location in locations:
            short_location = _truncate_location(ignored_reference_location)
            if short_location not in missing_reference_locations:
                msg = f'Reference {domain_type} {target} for {ignored_reference_location} can be removed from {app.config.missing_references_filename}. It is no longer a missing reference in the docs.'
                logger.warning(msg, location=ignored_reference_location, type='ref', subtype=domain_type)

def save_missing_references_handler(app, exc):
    if False:
        for i in range(10):
            print('nop')
    '\n    At the end of the sphinx build, check that all lines of the existing JSON\n    file are still necessary.\n\n    If the configuration value ``missing_references_write_json`` is set\n    then write a new JSON file containing missing references.\n    '
    if not app.config.missing_references_enabled:
        return
    _warn_unused_missing_references(app)
    json_path = Path(app.confdir) / app.config.missing_references_filename
    references_warnings = getattr(app.env, 'missing_references_warnings', {})
    if app.config.missing_references_write_json:
        _write_missing_references_json(references_warnings, json_path)

def _write_missing_references_json(records, json_path):
    if False:
        return 10
    "\n    Convert ignored references to a format which we can write as JSON\n\n    Convert from ``{(domain_type, target): locations}`` to\n    ``{domain_type: {target: locations}}`` since JSON can't serialize tuples.\n    "
    transformed_records = defaultdict(dict)
    for ((domain_type, target), paths) in records.items():
        transformed_records[domain_type][target] = sorted(paths)
    with json_path.open('w') as stream:
        json.dump(transformed_records, stream, sort_keys=True, indent=2)

def _read_missing_references_json(json_path):
    if False:
        while True:
            i = 10
    "\n    Convert from the JSON file to the form used internally by this\n    extension.\n\n    The JSON file is stored as ``{domain_type: {target: [locations,]}}``\n    since JSON can't store dictionary keys which are tuples. We convert\n    this back to ``{(domain_type, target):[locations]}`` for internal use.\n\n    "
    with json_path.open('r') as stream:
        data = json.load(stream)
    ignored_references = {}
    for (domain_type, targets) in data.items():
        for (target, locations) in targets.items():
            ignored_references[domain_type, target] = locations
    return ignored_references

def prepare_missing_references_handler(app):
    if False:
        print('Hello World!')
    '\n    Handler called to initialize this extension once the configuration\n    is ready.\n\n    Reads the missing references file and populates ``nitpick_ignore`` if\n    appropriate.\n    '
    if not app.config.missing_references_enabled:
        return
    sphinx_logger = logging.getLogger('sphinx')
    missing_reference_filter = MissingReferenceFilter(app)
    for handler in sphinx_logger.handlers:
        if isinstance(handler, sphinx_logging.WarningStreamHandler) and missing_reference_filter not in handler.filters:
            handler.filters.insert(0, missing_reference_filter)
    app.env.missing_references_ignored_references = {}
    json_path = Path(app.confdir) / app.config.missing_references_filename
    if not json_path.exists():
        return
    ignored_references = _read_missing_references_json(json_path)
    app.env.missing_references_ignored_references = ignored_references
    if not app.config.missing_references_write_json:
        app.config.nitpick_ignore = list(app.config.nitpick_ignore)
        app.config.nitpick_ignore.extend(ignored_references.keys())

def setup(app):
    if False:
        for i in range(10):
            print('nop')
    app.add_config_value('missing_references_enabled', True, 'env')
    app.add_config_value('missing_references_write_json', False, 'env')
    app.add_config_value('missing_references_warn_unused_ignores', True, 'env')
    app.add_config_value('missing_references_filename', 'missing-references.json', 'env')
    app.connect('builder-inited', prepare_missing_references_handler)
    app.connect('missing-reference', record_missing_reference_handler)
    app.connect('build-finished', save_missing_references_handler)
    return {'parallel_read_safe': True}