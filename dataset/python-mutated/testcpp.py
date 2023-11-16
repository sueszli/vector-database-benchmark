from unittest import TestCase, main
from multiprocessing import Process, Queue
from six.moves.queue import Empty
import sys
import locale
if '..' not in sys.path:
    sys.path.insert(0, '..')
from ply.lex import lex
from ply.cpp import *

def preprocessing(in_, out_queue):
    if False:
        for i in range(10):
            print('nop')
    out = None
    try:
        p = Preprocessor(lex())
        p.parse(in_)
        tokens = [t.value for t in p.parser]
        out = ''.join(tokens)
    finally:
        out_queue.put(out)

class CPPTests(TestCase):
    """Tests related to ANSI-C style lexical preprocessor."""

    def __test_preprocessing(self, in_, expected, time_limit=1.0):
        if False:
            while True:
                i = 10
        out_queue = Queue()
        preprocessor = Process(name='PLY`s C preprocessor', target=preprocessing, args=(in_, out_queue))
        preprocessor.start()
        try:
            out = out_queue.get(timeout=time_limit)
        except Empty:
            preprocessor.terminate()
            raise RuntimeError('Time limit exceeded!')
        else:
            self.assertMultiLineEqual(out, expected)

    def test_infinite_argument_expansion(self):
        if False:
            while True:
                i = 10
        self.__test_preprocessing('#define a(x) x\n#define b a(b)\nb\n', '\n\nb')

    def test_concatenation(self):
        if False:
            return 10
        self.__test_preprocessing('#define a(x) x##_\n#define b(x) _##x\n#define c(x) _##x##_\n#define d(x,y) _##x##y##_\n\na(i)\nb(j)\nc(k)\nd(q,s)', '\n\n\n\n\ni_\n_j\n_k_\n_qs_')

    def test_deadloop_macro(self):
        if False:
            i = 10
            return i + 15
        self.__test_preprocessing('#define a(x) x\n\na;', '\n\na;')

    def test_index_error(self):
        if False:
            while True:
                i = 10
        self.__test_preprocessing('#define a(x) x\n\na', '\n\na')

    def test_evalexpr(self):
        if False:
            for i in range(10):
                print('nop')
        self.__test_preprocessing('#if (1!=0) && (!x || (!(1==2)))\na;\n#else\nb;\n#endif\n', '\na;\n\n')

    def test_include_nonascii(self):
        if False:
            for i in range(10):
                print('nop')
        locale.setlocale(locale.LC_ALL, 'C')
        self.__test_preprocessing('#include "test_cpp_nonascii.c"\nx;\n\n', '\n \n1;\n')
main()