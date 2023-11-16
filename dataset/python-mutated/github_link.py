import inspect
import importlib
import os.path
from urlparse import urlunsplit
'\n* adds github_link(mode) context variable: provides URL (in relevant mode) of\n  current document on github\n* if sphinx.ext.linkcode is enabled, automatically generates github linkcode\n  links (by setting config.linkcode_resolve)\n\nSettings\n========\n\n* ``github_user``, username/organisation under which the project lives\n* ``github_project``, name of the project on github\n* (optional) ``version``, github branch to link to (default: master)\n\nNotes\n=====\n\n* provided ``linkcode_resolve`` only supports Python domain\n* generates https github links\n* explicitly imports ``odoo``, so useless for anyone else\n'

def setup(app):
    if False:
        i = 10
        return i + 15
    app.add_config_value('github_user', None, 'env')
    app.add_config_value('github_project', None, 'env')
    app.connect('html-page-context', add_doc_link)

    def linkcode_resolve(domain, info):
        if False:
            print('Hello World!')
        ' Resolves provided object to corresponding github URL\n        '
        if domain != 'py':
            return None
        if not (app.config.github_user and app.config.github_project):
            return None
        (module, fullname) = (info['module'], info['fullname'])
        if not module:
            return None
        obj = importlib.import_module(module)
        for item in fullname.split('.'):
            obj = getattr(obj, item, None)
        if obj is None:
            return None
        try:
            obj = getattr(obj, '_orig')
        except AttributeError:
            pass
        try:
            obj_source_path = inspect.getsourcefile(obj)
            (_, line) = inspect.getsourcelines(obj)
        except (TypeError, IOError):
            return None
        import odoo
        project_root = os.path.join(os.path.dirname(odoo.__file__), '..')
        return make_github_link(app, os.path.relpath(obj_source_path, project_root), line)
    app.config.linkcode_resolve = linkcode_resolve

def make_github_link(app, path, line=None, mode='blob'):
    if False:
        for i in range(10):
            print('nop')
    config = app.config
    urlpath = '/{user}/{project}/{mode}/{branch}/{path}'.format(user=config.github_user, project=config.github_project, branch=config.version or 'master', path=path, mode=mode)
    return urlunsplit(('https', 'github.com', urlpath, '', '' if line is None else 'L%d' % line))

def add_doc_link(app, pagename, templatename, context, doctree):
    if False:
        i = 10
        return i + 15
    ' Add github_link function linking to the current page on github '
    if not app.config.github_user and app.config.github_project:
        return
    source_suffix = app.config.source_suffix
    source_suffix = source_suffix if isinstance(source_suffix, basestring) else source_suffix[0]
    context['github_link'] = lambda mode='edit': make_github_link(app, 'doc/%s%s' % (pagename, source_suffix), mode=mode)