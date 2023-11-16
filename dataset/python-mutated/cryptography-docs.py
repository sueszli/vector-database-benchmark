from docutils import nodes
from docutils.parsers.rst import Directive
DANGER_MESSAGE = '\nThis is a "Hazardous Materials" module. You should **ONLY** use it if you\'re\n100% absolutely sure that you know what you\'re doing because this module is\nfull of land mines, dragons, and dinosaurs with laser guns.\n'
DANGER_ALTERNATE = '\n\nYou may instead be interested in :doc:`{alternate}`.\n'

class HazmatDirective(Directive):
    has_content = True

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        message = DANGER_MESSAGE
        if self.content:
            message += DANGER_ALTERNATE.format(alternate=self.content[0])
        content = nodes.paragraph('', message)
        admonition_node = Hazmat('\n'.join(content))
        self.state.nested_parse(content, self.content_offset, admonition_node)
        admonition_node.line = self.lineno
        return [admonition_node]

class Hazmat(nodes.Admonition, nodes.Element):
    pass

def html_visit_hazmat_node(self, node):
    if False:
        print('Hello World!')
    return self.visit_admonition(node, 'danger')

def latex_visit_hazmat_node(self, node):
    if False:
        for i in range(10):
            print('nop')
    return self.visit_admonition(node)

def depart_hazmat_node(self, node):
    if False:
        while True:
            i = 10
    return self.depart_admonition(node)

def setup(app):
    if False:
        for i in range(10):
            print('nop')
    app.add_node(Hazmat, html=(html_visit_hazmat_node, depart_hazmat_node), latex=(latex_visit_hazmat_node, depart_hazmat_node))
    app.add_directive('hazmat', HazmatDirective)
    return {'parallel_read_safe': True}