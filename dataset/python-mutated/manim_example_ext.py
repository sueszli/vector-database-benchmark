from docutils import nodes
from docutils.parsers.rst import directives, Directive
import jinja2
import os

class skip_manim_node(nodes.Admonition, nodes.Element):
    pass

def visit(self, node, name=''):
    if False:
        for i in range(10):
            print('nop')
    self.visit_admonition(node, name)

def depart(self, node):
    if False:
        print('Hello World!')
    self.depart_admonition(node)

class ManimExampleDirective(Directive):
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    option_spec = {'hide_code': bool, 'media': str}
    final_argument_whitespace = True

    def run(self):
        if False:
            print('Hello World!')
        hide_code = 'hide_code' in self.options
        scene_name = self.arguments[0]
        media_file_name = self.options['media']
        source_block = ['.. code-block:: python', '', *['    ' + line for line in self.content]]
        source_block = '\n'.join(source_block)
        state_machine = self.state_machine
        document = state_machine.document
        if any((media_file_name.endswith(ext) for ext in ['.png', '.jpg', '.gif'])):
            is_video = False
        else:
            is_video = True
        rendered_template = jinja2.Template(TEMPLATE).render(scene_name=scene_name, scene_name_lowercase=scene_name.lower(), hide_code=hide_code, is_video=is_video, media_file_name=media_file_name, source_block=source_block)
        state_machine.insert_input(rendered_template.split('\n'), source=document.attributes['source'])
        return []

def setup(app):
    if False:
        i = 10
        return i + 15
    app.add_node(skip_manim_node, html=(visit, depart))
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir
    app.add_directive('manim-example', ManimExampleDirective)
    metadata = {'parallel_read_safe': False, 'parallel_write_safe': True}
    return metadata
TEMPLATE = '\n{% if not hide_code %}\n\n.. raw:: html\n\n    <div class="manim-example">\n\n{% endif %}\n\n{% if is_video %}\n.. raw:: html\n\n    <video id="{{ scene_name_lowercase }}" class="manim-video" controls loop autoplay src="{{ media_file_name }}"></video>\n{% else %}\n.. image:: {{ media_file_name }}\n    :align: center\n    :name: {{ scene_name_lowercase }}\n{% endif %}\n\n{% if not hide_code %}\n.. raw:: html\n\n    <h5 class="example-header">{{ scene_name }}<a class="headerlink" href="#{{ scene_name_lowercase }}">¶</a></h5>\n\n{{ source_block }}\n{% endif %}\n\n.. raw:: html\n\n    </div>\n'