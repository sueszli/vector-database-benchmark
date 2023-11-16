import copy
from odoo.report.render.rml2pdf import utils

class odt2odt(object):

    def __init__(self, odt, localcontext):
        if False:
            for i in range(10):
                print('nop')
        self.localcontext = localcontext
        self.etree = odt
        self._node = None

    def render(self):
        if False:
            print('Hello World!')

        def process_text(node, new_node):
            if False:
                i = 10
                return i + 15
            for child in utils._child_get(node, self):
                new_child = copy.deepcopy(child)
                new_node.append(new_child)
                if len(child):
                    for n in new_child:
                        new_child.text = utils._process_text(self, child.text)
                        new_child.tail = utils._process_text(self, child.tail)
                        new_child.remove(n)
                    process_text(child, new_child)
                else:
                    new_child.text = utils._process_text(self, child.text)
                    new_child.tail = utils._process_text(self, child.tail)
        self._node = copy.deepcopy(self.etree)
        for n in self._node:
            self._node.remove(n)
        process_text(self.etree, self._node)
        return self._node

def parseNode(node, localcontext={}):
    if False:
        for i in range(10):
            print('nop')
    r = odt2odt(node, localcontext)
    return r.render()