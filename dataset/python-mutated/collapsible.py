from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.util.nodes import nested_parse_with_titles

class CollapsibleNode(nodes.Body, nodes.Element):

    def __init__(self, source, summary, *args, **kwargs):
        if False:
            return 10
        self.summary = summary
        super().__init__(source, *args, **kwargs)

def visit_collapsible_node(self, node):
    if False:
        while True:
            i = 10
    self.body.append('<details>\n')
    if node.summary:
        self.body.append(f'<summary>{node.summary}</summary>\n')

def depart_collapsible_node(self, node):
    if False:
        return 10
    self.body.append('</details>\n')

class CollapsibleSection(Directive):
    option_spec = {'summary': directives.unchanged}
    has_content = True

    def run(self):
        if False:
            while True:
                i = 10
        summary = self.options.get('summary', '')
        node = CollapsibleNode('\n'.join(self.content), summary)
        nested_parse_with_titles(self.state, self.content, node)
        return [node]