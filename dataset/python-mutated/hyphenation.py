from calibre.ebooks.oeb.base import OEB_DOCS
from polyglot.builtins import iteritems

def add_soft_hyphens(container, report=None):
    if False:
        print('Hello World!')
    from calibre.utils.hyphenation.hyphenate import add_soft_hyphens_to_html
    for (name, mt) in iteritems(container.mime_map):
        if mt not in OEB_DOCS:
            continue
        add_soft_hyphens_to_html(container.parsed(name), container.mi.language)
        container.dirty(name)
    if report is not None:
        report(_('Soft hyphens added'))

def remove_soft_hyphens(container, report=None):
    if False:
        return 10
    from calibre.utils.hyphenation.hyphenate import remove_soft_hyphens_from_html
    for (name, mt) in iteritems(container.mime_map):
        if mt not in OEB_DOCS:
            continue
        remove_soft_hyphens_from_html(container.parsed(name))
        container.dirty(name)
    if report is not None:
        report(_('Soft hyphens removed'))