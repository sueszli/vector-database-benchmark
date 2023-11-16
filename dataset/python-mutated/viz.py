from __future__ import annotations
from xml.etree import ElementTree as ET
from river.tree.base import Branch, Leaf

def tree_to_html(tree: Branch) -> ET.Element:
    if False:
        for i in range(10):
            print('nop')

    def add_node(node: Branch | Leaf, parent: ET.Element):
        if False:
            for i in range(10):
                print('nop')
        li = ET.Element('li')
        parent.append(li)
        code = ET.Element('code')
        if isinstance(node, Branch):
            code.text = node.repr_split
            li.append(code)
            ul = ET.Element('ul')
            for child in node.children:
                add_node(node=child, parent=ul)
            li.append(ul)
        else:
            code.text = repr(node)
            li.append(code)
    root = ET.Element('ul', attrib={'class': 'tree'})
    add_node(node=tree, parent=root)
    return root
CSS = '\n.tree,\n.tree ul,\n.tree li {\n    list-style: none;\n    margin: 0;\n    padding: 0;\n    position: relative;\n}\n\n.tree {\n    margin: 0 0 1em;\n    text-align: center;\n}\n\n.tree,\n.tree ul {\n    display: table;\n}\n\n.tree ul {\n    width: 100%;\n}\n\n.tree li {\n    display: table-cell;\n    padding: .5em 0;\n    vertical-align: top;\n}\n\n.tree li:before {\n    outline: solid 1px #666;\n    content: "";\n    left: 0;\n    position: absolute;\n    right: 0;\n    top: 0;\n}\n\n.tree li:first-child:before {\n    left: 50%;\n}\n\n.tree li:last-child:before {\n    right: 50%;\n}\n\n.tree code,\n.tree span {\n    border: solid .1em #666;\n    display: inline-block;\n    margin: 0 .2em .5em;\n    padding: .2em .5em;\n    position: relative;\n}\n\n.tree ul:before,\n.tree code:before,\n.tree span:before {\n    outline: solid 1px #666;\n    content: "";\n    height: .5em;\n    left: 50%;\n    position: absolute;\n}\n\n.tree ul:before {\n    top: -.5em;\n}\n\n.tree code:before,\n.tree span:before {\n    top: -.55em;\n}\n\n.tree>li {\n    margin-top: 0;\n}\n\n.tree>li:before,\n.tree>li:after,\n.tree>li>code:before,\n.tree>li>span:before {\n    outline: none;\n}\n'