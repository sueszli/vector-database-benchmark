"""Allow `MathJax`_ to be used to display math in Sphinx's HTML writer.

This requires the MathJax JavaScript library on your webserver/computer.

.. _MathJax: https://www.mathjax.org/
"""
from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, cast
from docutils import nodes
import sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.domains.math import MathDomain
from sphinx.errors import ExtensionError
from sphinx.locale import _
from sphinx.util.math import get_node_equation_number
if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.writers.html import HTML5Translator
MATHJAX_URL = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
logger = sphinx.util.logging.getLogger(__name__)

def html_visit_math(self: HTML5Translator, node: nodes.math) -> None:
    if False:
        for i in range(10):
            print('nop')
    self.body.append(self.starttag(node, 'span', '', CLASS='math notranslate nohighlight'))
    self.body.append(self.builder.config.mathjax_inline[0] + self.encode(node.astext()) + self.builder.config.mathjax_inline[1] + '</span>')
    raise nodes.SkipNode

def html_visit_displaymath(self: HTML5Translator, node: nodes.math_block) -> None:
    if False:
        return 10
    self.body.append(self.starttag(node, 'div', CLASS='math notranslate nohighlight'))
    if node['nowrap']:
        self.body.append(self.encode(node.astext()))
        self.body.append('</div>')
        raise nodes.SkipNode
    if node['number']:
        number = get_node_equation_number(self, node)
        self.body.append('<span class="eqno">(%s)' % number)
        self.add_permalink_ref(node, _('Link to this equation'))
        self.body.append('</span>')
    self.body.append(self.builder.config.mathjax_display[0])
    parts = [prt for prt in node.astext().split('\n\n') if prt.strip()]
    if len(parts) > 1:
        self.body.append(' \\begin{align}\\begin{aligned}')
    for (i, part) in enumerate(parts):
        part = self.encode(part)
        if '\\\\' in part:
            self.body.append('\\begin{split}' + part + '\\end{split}')
        else:
            self.body.append(part)
        if i < len(parts) - 1:
            self.body.append('\\\\')
    if len(parts) > 1:
        self.body.append('\\end{aligned}\\end{align} ')
    self.body.append(self.builder.config.mathjax_display[1])
    self.body.append('</div>\n')
    raise nodes.SkipNode

def install_mathjax(app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], event_arg: Any) -> None:
    if False:
        print('Hello World!')
    if app.builder.format != 'html' or app.builder.math_renderer_name != 'mathjax':
        return
    if not app.config.mathjax_path:
        msg = 'mathjax_path config value must be set for the mathjax extension to work'
        raise ExtensionError(msg)
    domain = cast(MathDomain, app.env.get_domain('math'))
    builder = cast(StandaloneHTMLBuilder, app.builder)
    if app.registry.html_assets_policy == 'always' or domain.has_equations(pagename):
        if app.config.mathjax2_config:
            if app.config.mathjax_path == MATHJAX_URL:
                logger.warning('mathjax_config/mathjax2_config does not work for the current MathJax version, use mathjax3_config instead')
            body = 'MathJax.Hub.Config(%s)' % json.dumps(app.config.mathjax2_config)
            builder.add_js_file('', type='text/x-mathjax-config', body=body)
        if app.config.mathjax3_config:
            body = 'window.MathJax = %s' % json.dumps(app.config.mathjax3_config)
            builder.add_js_file('', body=body)
        options = {}
        if app.config.mathjax_options:
            options.update(app.config.mathjax_options)
        if 'async' not in options and 'defer' not in options:
            if app.config.mathjax3_config:
                options['defer'] = 'defer'
            else:
                options['async'] = 'async'
        builder.add_js_file(app.config.mathjax_path, **options)

def setup(app: Sphinx) -> dict[str, Any]:
    if False:
        while True:
            i = 10
    app.add_html_math_renderer('mathjax', (html_visit_math, None), (html_visit_displaymath, None))
    app.add_config_value('mathjax_path', MATHJAX_URL, 'html')
    app.add_config_value('mathjax_options', {}, 'html')
    app.add_config_value('mathjax_inline', ['\\(', '\\)'], 'html')
    app.add_config_value('mathjax_display', ['\\[', '\\]'], 'html')
    app.add_config_value('mathjax_config', None, 'html')
    app.add_config_value('mathjax2_config', lambda c: c.mathjax_config, 'html')
    app.add_config_value('mathjax3_config', None, 'html')
    app.connect('html-page-context', install_mathjax)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}