"""
Extension for building Qt-like documentation.

 - Method lists preceding the actual method documentation
 - Inherited members documented separately
 - Members inherited from Qt have links to qt-project documentation
 - Signal documentation

"""

def setup(app):
    if False:
        i = 10
        return i + 15
    app.setup_extension('sphinx.ext.autodoc')
    app.add_config_value('todo_include_todos', False, False)
    app.add_node(Todolist)
    app.add_node(Todo, html=(visit_todo_node, depart_todo_node), latex=(visit_todo_node, depart_todo_node), text=(visit_todo_node, depart_todo_node))
    app.add_directive('todo', TodoDirective)
    app.add_directive('todolist', TodolistDirective)
    app.connect('doctree-resolved', process_todo_nodes)
    app.connect('env-purge-doc', purge_todos)
from docutils import nodes
from sphinx.util.compat import Directive
from sphinx.util.compat import make_admonition

class Todolist(nodes.General, nodes.Element):
    pass

class TodolistDirective(Directive):

    def run(self):
        if False:
            i = 10
            return i + 15
        return [Todolist('')]

class Todo(nodes.Admonition, nodes.Element):
    pass

def visit_todo_node(self, node):
    if False:
        i = 10
        return i + 15
    self.visit_admonition(node)

def depart_todo_node(self, node):
    if False:
        return 10
    self.depart_admonition(node)

class TodoDirective(Directive):
    has_content = True

    def run(self):
        if False:
            while True:
                i = 10
        env = self.state.document.settings.env
        targetid = 'todo-%d' % env.new_serialno('todo')
        targetnode = nodes.target('', '', ids=[targetid])
        ad = make_admonition(Todo, self.name, ['Todo'], self.options, self.content, self.lineno, self.content_offset, self.block_text, self.state, self.state_machine)
        if not hasattr(env, 'todo_all_todos'):
            env.todo_all_todos = []
        env.todo_all_todos.append({'docname': env.docname, 'lineno': self.lineno, 'todo': ad[0].deepcopy(), 'target': targetnode})
        return [targetnode] + ad

def purge_todos(app, env, docname):
    if False:
        print('Hello World!')
    if not hasattr(env, 'todo_all_todos'):
        return
    env.todo_all_todos = [todo for todo in env.todo_all_todos if todo['docname'] != docname]

def process_todo_nodes(app, doctree, fromdocname):
    if False:
        for i in range(10):
            print('nop')
    if not app.config.todo_include_todos:
        for node in doctree.traverse(Todo):
            node.parent.remove(node)
    env = app.builder.env
    for node in doctree.traverse(Todolist):
        if not app.config.todo_include_todos:
            node.replace_self([])
            continue
        content = []
        for todo_info in env.todo_all_todos:
            para = nodes.paragraph()
            filename = env.doc2path(todo_info['docname'], base=None)
            description = '(The original entry is located in %s, line %d and can be found ' % (filename, todo_info['lineno'])
            para += nodes.Text(description, description)
            newnode = nodes.reference('', '')
            innernode = nodes.emphasis('here', 'here')
            newnode['refdocname'] = todo_info['docname']
            newnode['refuri'] = app.builder.get_relative_uri(fromdocname, todo_info['docname'])
            newnode['refuri'] += '#' + todo_info['target']['refid']
            newnode.append(innernode)
            para += newnode
            para += nodes.Text('.)', '.)')
            content.append(todo_info['todo'])
            content.append(para)
        node.replace_self(content)