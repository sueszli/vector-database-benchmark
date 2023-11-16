"""
A Sphinx Extension for Scapy's doc preprocessing
"""
import subprocess
import os
from scapy.packet import Packet, _pkt_ls, rfc
from sphinx.ext.autodoc import AttributeDocumenter

def generate_rest_table(items):
    if False:
        print('Hello World!')
    '\n    Generates a ReST table from a list of tuples\n    '
    lengths = [max((len(y) for y in x)) for x in zip(*items)]
    sep = '+%s+' % '+'.join(('-' * x for x in lengths))
    sized = '|%s|' % '|'.join(('{:%ss}' % x for x in lengths))
    output = []
    for i in items:
        output.append(sep)
        output.append(sized.format(*i))
    output.append(sep)
    return output

def tab(items):
    if False:
        i = 10
        return i + 15
    '\n    Tabulize a generator.\n    '
    for i in items:
        yield ('   ' + i)

def class_ref(cls):
    if False:
        i = 10
        return i + 15
    '\n    Get Sphinx reference to a class\n    '
    return ':class:`~%s`' % (cls.__module__ + '.' + cls.__name__)

def get_fields_desc(obj):
    if False:
        return 10
    '\n    Create a readable documentation for fields_desc\n    '
    output = []
    for value in _pkt_ls(obj):
        (fname, cls, clsne, dflt, long_attrs) = value
        output.append(('**%s**' % fname, class_ref(cls) + (' ' + clsne if clsne else ''), '``%s``' % dflt))
    if output:
        output = list(tab(generate_rest_table(output)))
        output.insert(0, '.. table:: %s fields' % obj.__name__)
        output.insert(1, '   :widths: grid')
        output.insert(2, '   ')
        try:
            graph = list(tab(rfc(obj, ret=True).split('\n')))
        except AttributeError:
            return output
        s = 'Display RFC-like schema'
        graph.insert(0, '.. raw:: html')
        graph.insert(1, '')
        graph.insert(2, '   <details><summary>%s</summary><code><pre>' % s)
        graph.append('   </pre></code></details>')
        graph.append('')
        return graph + output
    return output

class AttrsDocumenter(AttributeDocumenter):
    """
    Mock of AttributeDocumenter to handle Scapy settings
    """

    def add_directive_header(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15

        def call_parent():
            if False:
                i = 10
                return i + 15
            'Calls the super.super.add_directive_header'
            super(AttributeDocumenter, self).add_directive_header(*args, **kwargs)
        sourcename = self.get_sourcename()
        if issubclass(self.parent, Packet):
            if self.object_name == 'fields_desc':
                call_parent()
                table = list(tab(get_fields_desc(self.parent)))
                if table:
                    self.add_line('   ', sourcename)
                    for line in table:
                        self.add_line(line, sourcename)
                    self.add_line('   ', sourcename)
                return
            elif self.object_name == 'payload_guess':
                call_parent()
                children = sorted(set((class_ref(x[1]) for x in self.object)))
                if children:
                    lines = ['', 'Possible sublayers:', ', '.join(children), '']
                    for line in tab(lines):
                        self.add_line(line, sourcename)
                return
            elif self.object_name in ['aliastypes'] or self.object_name.startswith('class_'):
                call_parent()
                return
        super(AttrsDocumenter, self).add_directive_header(*args, **kwargs)

def builder_inited_handler(app):
    if False:
        for i in range(10):
            print('nop')
    'Generate API tree'
    if int(os.environ.get('SCAPY_APITREE', True)):
        subprocess.call(['tox', '-e', 'apitree'])

def setup(app):
    if False:
        i = 10
        return i + 15
    '\n    Entry point of the scapy_doc extension.\n\n    Called by sphinx while booting up.\n    '
    app.add_autodocumenter(AttrsDocumenter, override=True)
    app.connect('builder-inited', builder_inited_handler)
    return {'version': '1.0', 'parallel_read_safe': True, 'parallel_write_safe': True}