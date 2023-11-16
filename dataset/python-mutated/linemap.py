from uncompyle6.semantics.pysource import SourceWalker, code_deparse
import uncompyle6.semantics.fragments as fragments

class LineMapWalker(SourceWalker):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(LineMapWalker, self).__init__(*args, **kwargs)
        self.source_linemap = {}
        self.current_line_number = 1

    def write(self, *data):
        if False:
            while True:
                i = 10
        'Augment write routine to keep track of current line'
        for l in data:
            for i in str(l):
                if i == '\n':
                    self.current_line_number += 1
                    pass
                pass
            pass
        return super(LineMapWalker, self).write(*data)

    def default(self, node):
        if False:
            i = 10
            return i + 15
        'Augment write default routine to record line number changes'
        if hasattr(node, 'linestart'):
            if node.linestart:
                self.source_linemap[self.current_line_number] = node.linestart
        return super(LineMapWalker, self).default(node)

    def n_LOAD_CONST(self, node):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(node, 'linestart'):
            if node.linestart:
                self.source_linemap[self.current_line_number] = node.linestart
        return super(LineMapWalker, self).n_LOAD_CONST(node)

class LineMapFragmentWalker(fragments.FragmentsWalker, LineMapWalker):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(LineMapFragmentWalker, self).__init__(*args, **kwargs)
        self.source_linemap = {}
        self.current_line_number = 0

def deparse_code_with_map(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Like deparse_code but saves line number correspondences.\n    Deprecated. Use code_deparse_with_map\n    '
    kwargs['walker'] = LineMapWalker
    return code_deparse(*args, **kwargs)

def code_deparse_with_map(*args, **kwargs):
    if False:
        print('Hello World!')
    '\n    Like code_deparse but saves line number correspondences.\n    '
    kwargs['walker'] = LineMapWalker
    return code_deparse(*args, **kwargs)

def deparse_code_with_fragments_and_map(*args, **kwargs):
    if False:
        print('Hello World!')
    '\n    Like deparse_code_with_map but saves fragments.\n    Deprecated. Use code_deparse_with_fragments_and_map\n    '
    kwargs['walker'] = LineMapFragmentWalker
    return fragments.deparse_code(*args, **kwargs)

def code_deparse_with_fragments_and_map(*args, **kwargs):
    if False:
        print('Hello World!')
    '\n    Like code_deparse_with_map but saves fragments.\n    '
    kwargs['walker'] = LineMapFragmentWalker
    return fragments.code_deparse(*args, **kwargs)
if __name__ == '__main__':

    def deparse_test(co):
        if False:
            return 10
        'This is a docstring'
        deparsed = code_deparse_with_map(co)
        a = 1
        b = 2
        print('\n')
        linemap = [(line_no, deparsed.source_linemap[line_no]) for line_no in sorted(deparsed.source_linemap.keys())]
        print(linemap)
        deparsed = code_deparse_with_fragments_and_map(co)
        print('\n')
        linemap2 = [(line_no, deparsed.source_linemap[line_no]) for line_no in sorted(deparsed.source_linemap.keys())]
        print(linemap2)
        return
    deparse_test(deparse_test.__code__)