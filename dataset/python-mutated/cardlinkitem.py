"""
Directive "cardlinkitem" used in tutorials navigation page.
"""
import os
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from docutils import nodes
from sphinx.addnodes import pending_xref
TAG_TEMPLATE = '<span class="card-link-tag">{tag}</span>'
TAGS_TEMPLATE = '\n    <p class="card-link-summary">{tags}</p>\n'
CARD_HEADER = '\n.. raw:: html\n\n    <div class="card-link admonition">\n\n    <a class="card-link-clickable" href="#">\n\n    <div class="card-link-body">\n\n    <div class="card-link-text">\n\n    <div class="card-link-title-container">\n        <h4>{header}</h4>\n    </div>\n\n    <p class="card-link-summary">{description}</p>\n\n    {tags}\n\n    </div>\n\n    <div class="card-link-icon circle {image_background}">\n\n.. image:: {image}\n\n.. raw:: html\n\n    </div>\n\n    </div>\n\n    </a>\n'
CARD_FOOTER = '\n.. raw:: html\n\n    </div>\n'

class CustomCardItemDirective(Directive):
    option_spec = {'header': directives.unchanged, 'image': directives.unchanged, 'background': directives.unchanged, 'link': directives.unchanged, 'description': directives.unchanged, 'tags': directives.unchanged}

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        env = self.state.document.settings.env
        try:
            if 'header' in self.options:
                header = self.options['header']
            else:
                raise ValueError('header not found')
            if 'link' in self.options:
                link = directives.uri(self.options['link'])
            else:
                raise ValueError('link not found')
            if 'image' in self.options:
                image = directives.uri(self.options['image'])
            else:
                image = os.path.join(os.path.relpath(env.app.srcdir, env.app.confdir), '../img/thumbnails/nni_icon_white.png')
            image_background = self.options.get('background', 'indigo')
            description = self.options.get('description', '')
            tags = self.options.get('tags', '').strip().split('/')
            tags = [t.strip() for t in tags if t.strip()]
        except ValueError as e:
            print(e)
            raise
        if tags:
            tags_rst = TAGS_TEMPLATE.format(tags=''.join([TAG_TEMPLATE.format(tag=tag) for tag in tags]))
        else:
            tags_rst = ''
        card_rst = CARD_HEADER.format(header=header, image=image, image_background=image_background, link=link, description=description, tags=tags_rst)
        card = nodes.paragraph()
        self.state.nested_parse(StringList(card_rst.split('\n')), self.content_offset, card)
        link_node = pending_xref('<a/>', reftype='doc', refdomain='std', reftarget=link, refexplicit=False, refwarn=True)
        link_node += nodes.paragraph(header)
        link_node['classes'] = ['card-link-anchor']
        card += link_node
        self.state.nested_parse(StringList(CARD_FOOTER.split('\n')), self.content_offset, card)
        return [card]

def setup(app):
    if False:
        print('Hello World!')
    app.add_directive('cardlinkitem', CustomCardItemDirective)