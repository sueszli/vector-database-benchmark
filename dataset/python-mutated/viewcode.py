"""Add links to module code in Python object descriptions."""
from __future__ import annotations
import posixpath
import traceback
from importlib import import_module
from os import path
from typing import TYPE_CHECKING, Any, cast
from docutils import nodes
from docutils.nodes import Element, Node
import sphinx
from sphinx import addnodes
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import logging
from sphinx.util.display import status_iterator
from sphinx.util.nodes import make_refnode
if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from sphinx.application import Sphinx
    from sphinx.builders import Builder
    from sphinx.environment import BuildEnvironment
logger = logging.getLogger(__name__)
OUTPUT_DIRNAME = '_modules'

class viewcode_anchor(Element):
    """Node for viewcode anchors.

    This node will be processed in the resolving phase.
    For viewcode supported builders, they will be all converted to the anchors.
    For not supported builders, they will be removed.
    """

def _get_full_modname(modname: str, attribute: str) -> str | None:
    if False:
        while True:
            i = 10
    try:
        if modname is None:
            return None
        module = import_module(modname)
        value = module
        for attr in attribute.split('.'):
            if attr:
                value = getattr(value, attr)
        return getattr(value, '__module__', None)
    except AttributeError:
        logger.verbose("Didn't find %s in %s", attribute, modname)
        return None
    except Exception as e:
        logger.verbose(traceback.format_exc().rstrip())
        logger.verbose('viewcode can\'t import %s, failed with error "%s"', modname, e)
        return None

def is_supported_builder(builder: Builder) -> bool:
    if False:
        print('Hello World!')
    if builder.format != 'html':
        return False
    if builder.name == 'singlehtml':
        return False
    if builder.name.startswith('epub') and (not builder.config.viewcode_enable_epub):
        return False
    return True

def doctree_read(app: Sphinx, doctree: Node) -> None:
    if False:
        for i in range(10):
            print('nop')
    env = app.builder.env
    if not hasattr(env, '_viewcode_modules'):
        env._viewcode_modules = {}

    def has_tag(modname: str, fullname: str, docname: str, refname: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        entry = env._viewcode_modules.get(modname, None)
        if entry is False:
            return False
        code_tags = app.emit_firstresult('viewcode-find-source', modname)
        if code_tags is None:
            try:
                analyzer = ModuleAnalyzer.for_module(modname)
                analyzer.find_tags()
            except Exception:
                env._viewcode_modules[modname] = False
                return False
            code = analyzer.code
            tags = analyzer.tags
        else:
            (code, tags) = code_tags
        if entry is None or entry[0] != code:
            entry = (code, tags, {}, refname)
            env._viewcode_modules[modname] = entry
        (_, tags, used, _) = entry
        if fullname in tags:
            used[fullname] = docname
            return True
        return False
    for objnode in list(doctree.findall(addnodes.desc)):
        if objnode.get('domain') != 'py':
            continue
        names: set[str] = set()
        for signode in objnode:
            if not isinstance(signode, addnodes.desc_signature):
                continue
            modname = signode.get('module')
            fullname = signode.get('fullname')
            refname = modname
            if env.config.viewcode_follow_imported_members:
                new_modname = app.emit_firstresult('viewcode-follow-imported', modname, fullname)
                if not new_modname:
                    new_modname = _get_full_modname(modname, fullname)
                modname = new_modname
            if not modname:
                continue
            fullname = signode.get('fullname')
            if not has_tag(modname, fullname, env.docname, refname):
                continue
            if fullname in names:
                continue
            names.add(fullname)
            pagename = posixpath.join(OUTPUT_DIRNAME, modname.replace('.', '/'))
            signode += viewcode_anchor(reftarget=pagename, refid=fullname, refdoc=env.docname)

def env_merge_info(app: Sphinx, env: BuildEnvironment, docnames: Iterable[str], other: BuildEnvironment) -> None:
    if False:
        return 10
    if not hasattr(other, '_viewcode_modules'):
        return
    if not hasattr(env, '_viewcode_modules'):
        env._viewcode_modules = {}
    for (modname, entry) in other._viewcode_modules.items():
        if modname not in env._viewcode_modules:
            env._viewcode_modules[modname] = entry
        elif env._viewcode_modules[modname]:
            used = env._viewcode_modules[modname][2]
            for (fullname, docname) in entry[2].items():
                if fullname not in used:
                    used[fullname] = docname

def env_purge_doc(app: Sphinx, env: BuildEnvironment, docname: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    modules = getattr(env, '_viewcode_modules', {})
    for (modname, entry) in list(modules.items()):
        if entry is False:
            continue
        (code, tags, used, refname) = entry
        for fullname in list(used):
            if used[fullname] == docname:
                used.pop(fullname)
        if len(used) == 0:
            modules.pop(modname)

class ViewcodeAnchorTransform(SphinxPostTransform):
    """Convert or remove viewcode_anchor nodes depends on builder."""
    default_priority = 100

    def run(self, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        if is_supported_builder(self.app.builder):
            self.convert_viewcode_anchors()
        else:
            self.remove_viewcode_anchors()

    def convert_viewcode_anchors(self) -> None:
        if False:
            i = 10
            return i + 15
        for node in self.document.findall(viewcode_anchor):
            anchor = nodes.inline('', _('[source]'), classes=['viewcode-link'])
            refnode = make_refnode(self.app.builder, node['refdoc'], node['reftarget'], node['refid'], anchor)
            node.replace_self(refnode)

    def remove_viewcode_anchors(self) -> None:
        if False:
            return 10
        for node in list(self.document.findall(viewcode_anchor)):
            node.parent.remove(node)

def get_module_filename(app: Sphinx, modname: str) -> str | None:
    if False:
        while True:
            i = 10
    'Get module filename for *modname*.'
    source_info = app.emit_firstresult('viewcode-find-source', modname)
    if source_info:
        return None
    else:
        try:
            (filename, source) = ModuleAnalyzer.get_module_source(modname)
            return filename
        except Exception:
            return None

def should_generate_module_page(app: Sphinx, modname: str) -> bool:
    if False:
        while True:
            i = 10
    'Check generation of module page is needed.'
    module_filename = get_module_filename(app, modname)
    if module_filename is None:
        return True
    builder = cast(StandaloneHTMLBuilder, app.builder)
    basename = modname.replace('.', '/') + builder.out_suffix
    page_filename = path.join(app.outdir, '_modules/', basename)
    try:
        if path.getmtime(module_filename) <= path.getmtime(page_filename):
            return False
    except OSError:
        pass
    return True

def collect_pages(app: Sphinx) -> Generator[tuple[str, dict[str, Any], str], None, None]:
    if False:
        while True:
            i = 10
    env = app.builder.env
    if not hasattr(env, '_viewcode_modules'):
        return
    if not is_supported_builder(app.builder):
        return
    highlighter = app.builder.highlighter
    urito = app.builder.get_relative_uri
    modnames = set(env._viewcode_modules)
    for (modname, entry) in status_iterator(sorted(env._viewcode_modules.items()), __('highlighting module code... '), 'blue', len(env._viewcode_modules), app.verbosity, lambda x: x[0]):
        if not entry:
            continue
        if not should_generate_module_page(app, modname):
            continue
        (code, tags, used, refname) = entry
        pagename = posixpath.join(OUTPUT_DIRNAME, modname.replace('.', '/'))
        if env.config.highlight_language in {'default', 'none'}:
            lexer = env.config.highlight_language
        else:
            lexer = 'python'
        linenos = 'inline' * env.config.viewcode_line_numbers
        highlighted = highlighter.highlight_block(code, lexer, linenos=linenos)
        lines = highlighted.splitlines()
        (before, after) = lines[0].split('<pre>')
        lines[0:1] = [before + '<pre>', after]
        max_index = len(lines) - 1
        link_text = _('[docs]')
        for (name, docname) in used.items():
            (type, start, end) = tags[name]
            backlink = urito(pagename, docname) + '#' + refname + '.' + name
            lines[start] = f'<div class="viewcode-block" id="{name}">\n<a class="viewcode-back" href="{backlink}">{link_text}</a>\n' + lines[start]
            lines[min(end, max_index)] += '</div>\n'
        parents = []
        parent = modname
        while '.' in parent:
            parent = parent.rsplit('.', 1)[0]
            if parent in modnames:
                parents.append({'link': urito(pagename, posixpath.join(OUTPUT_DIRNAME, parent.replace('.', '/'))), 'title': parent})
        parents.append({'link': urito(pagename, posixpath.join(OUTPUT_DIRNAME, 'index')), 'title': _('Module code')})
        parents.reverse()
        context = {'parents': parents, 'title': modname, 'body': _('<h1>Source code for %s</h1>') % modname + '\n'.join(lines)}
        yield (pagename, context, 'page.html')
    if not modnames:
        return
    html = ['\n']
    stack = ['']
    for modname in sorted(modnames):
        if modname.startswith(stack[-1]):
            stack.append(modname + '.')
            html.append('<ul>')
        else:
            stack.pop()
            while not modname.startswith(stack[-1]):
                stack.pop()
                html.append('</ul>')
            stack.append(modname + '.')
        relative_uri = urito(posixpath.join(OUTPUT_DIRNAME, 'index'), posixpath.join(OUTPUT_DIRNAME, modname.replace('.', '/')))
        html.append(f'<li><a href="{relative_uri}">{modname}</a></li>\n')
    html.append('</ul>' * (len(stack) - 1))
    context = {'title': _('Overview: module code'), 'body': _('<h1>All modules for which code is available</h1>') + ''.join(html)}
    yield (posixpath.join(OUTPUT_DIRNAME, 'index'), context, 'page.html')

def setup(app: Sphinx) -> dict[str, Any]:
    if False:
        return 10
    app.add_config_value('viewcode_import', None, False)
    app.add_config_value('viewcode_enable_epub', False, False)
    app.add_config_value('viewcode_follow_imported_members', True, False)
    app.add_config_value('viewcode_line_numbers', False, 'env', (bool,))
    app.connect('doctree-read', doctree_read)
    app.connect('env-merge-info', env_merge_info)
    app.connect('env-purge-doc', env_purge_doc)
    app.connect('html-collect-pages', collect_pages)
    app.add_event('viewcode-find-source')
    app.add_event('viewcode-follow-imported')
    app.add_post_transform(ViewcodeAnchorTransform)
    return {'version': sphinx.__display_version__, 'env_version': 1, 'parallel_read_safe': True}