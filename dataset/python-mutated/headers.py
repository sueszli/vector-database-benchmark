from ext.utils import get_sectionname, isections
from ext.indexer import tour_descinfo
import os

def setup(app):
    if False:
        return 10
    app.setup_extension('ext.indexer')
    app.add_config_value('headers_dest', '.', 'html')
    app.add_config_value('headers_mkdirs', False, '')
    app.add_config_value('headers_filename_sfx', '', 'html')
    app.add_config_value('headers_template', 'header.h', 'html')
    app.connect('html-page-context', writer)

def writer(app, pagename, templatename, context, doctree):
    if False:
        i = 10
        return i + 15
    if doctree is None:
        return
    env = app.builder.env
    dirpath = os.path.abspath(app.config['headers_dest'])
    if app.config['headers_mkdirs'] and (not os.path.lexists(dirpath)):
        os.makedirs(dirpath)
    filename_suffix = app.config['headers_filename_sfx']
    items = []
    for section in isections(doctree):
        tour_descinfo(items.append, section, env)
    if not items:
        return
    templates = app.builder.templates
    filename = f'{os.path.basename(pagename)}{filename_suffix}.h'
    filepath = os.path.join(dirpath, filename)
    template = app.config['headers_template']
    header = open(filepath, 'w', encoding='utf-8')
    context['hdr_items'] = items
    try:
        header.write(templates.render(template, context))
    finally:
        header.close()
        del context['hdr_items']