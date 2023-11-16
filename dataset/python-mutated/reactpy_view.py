import os
from typing import Any, ClassVar
from docs_app.examples import get_normalized_example_name
from docutils.nodes import raw
from docutils.parsers.rst import directives
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
_REACTPY_EXAMPLE_HOST = os.environ.get('REACTPY_DOC_EXAMPLE_SERVER_HOST', '')
_REACTPY_STATIC_HOST = os.environ.get('REACTPY_DOC_STATIC_SERVER_HOST', '/docs').rstrip('/')

class IteractiveWidget(SphinxDirective):
    has_content = False
    required_arguments = 1
    _next_id = 0
    option_spec: ClassVar[dict[str, Any]] = {'activate-button': directives.flag, 'margin': float}

    def run(self):
        if False:
            return 10
        IteractiveWidget._next_id += 1
        container_id = f'reactpy-widget-{IteractiveWidget._next_id}'
        view_id = get_normalized_example_name(self.arguments[0], self.get_source_info()[0])
        return [raw('', f'''\n                <div>\n                    <div\n                        id="{container_id}"\n                        class="interactive widget-container"\n                        style="margin-bottom: {self.options.get('margin', 0)}px;"\n                    />\n                    <script type="module">\n                        import {{ mountWidgetExample }} from "{_REACTPY_STATIC_HOST}/_static/custom.js";\n                        mountWidgetExample(\n                            "{container_id}",\n                            "{view_id}",\n                            "{_REACTPY_EXAMPLE_HOST}",\n                            {('true' if 'activate-button' in self.options else 'false')},\n                        );\n                    </script>\n                </div>\n                ''', format='html')]

def setup(app: Sphinx) -> None:
    if False:
        while True:
            i = 10
    app.add_directive('reactpy-view', IteractiveWidget)