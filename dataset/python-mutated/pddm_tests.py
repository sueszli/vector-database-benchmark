"""Tests for pddm.py."""
import io
import unittest
import pddm

class TestParsingMacros(unittest.TestCase):

    def testParseEmpty(self):
        if False:
            return 10
        f = io.StringIO(u'')
        result = pddm.MacroCollection(f)
        self.assertEqual(len(result._macros), 0)

    def testParseOne(self):
        if False:
            for i in range(10):
                print('nop')
        f = io.StringIO(u'PDDM-DEFINE foo( )\nbody')
        result = pddm.MacroCollection(f)
        self.assertEqual(len(result._macros), 1)
        macro = result._macros.get('foo')
        self.assertIsNotNone(macro)
        self.assertEquals(macro.name, 'foo')
        self.assertEquals(macro.args, tuple())
        self.assertEquals(macro.body, 'body')

    def testParseGeneral(self):
        if False:
            while True:
                i = 10
        f = io.StringIO(u'\nPDDM-DEFINE noArgs( )\nbody1\nbody2\n\nPDDM-DEFINE-END\n\nPDDM-DEFINE oneArg(foo)\nbody3\nPDDM-DEFINE  twoArgs( bar_ , baz )\nbody4\nbody5')
        result = pddm.MacroCollection(f)
        self.assertEqual(len(result._macros), 3)
        macro = result._macros.get('noArgs')
        self.assertIsNotNone(macro)
        self.assertEquals(macro.name, 'noArgs')
        self.assertEquals(macro.args, tuple())
        self.assertEquals(macro.body, 'body1\nbody2\n')
        macro = result._macros.get('oneArg')
        self.assertIsNotNone(macro)
        self.assertEquals(macro.name, 'oneArg')
        self.assertEquals(macro.args, ('foo',))
        self.assertEquals(macro.body, 'body3')
        macro = result._macros.get('twoArgs')
        self.assertIsNotNone(macro)
        self.assertEquals(macro.name, 'twoArgs')
        self.assertEquals(macro.args, ('bar_', 'baz'))
        self.assertEquals(macro.body, 'body4\nbody5')
        f = io.StringIO(u'\nPDDM-DEFINE another(a,b,c)\nbody1\nbody2')
        result.ParseInput(f)
        self.assertEqual(len(result._macros), 4)
        macro = result._macros.get('another')
        self.assertIsNotNone(macro)
        self.assertEquals(macro.name, 'another')
        self.assertEquals(macro.args, ('a', 'b', 'c'))
        self.assertEquals(macro.body, 'body1\nbody2')

    def testParseDirectiveIssues(self):
        if False:
            return 10
        test_list = [(u'PDDM-DEFINE foo()\nbody\nPDDM-DEFINED foo\nbaz', 'Hit a line with an unknown directive: '), (u'PDDM-DEFINE foo()\nbody\nPDDM-DEFINE-END\nPDDM-DEFINE-END\n', 'Got DEFINE-END directive without an active macro: '), (u'PDDM-DEFINE foo()\nbody\nPDDM-DEFINE-END\nmumble\n', "Hit a line that wasn't a directive and no open macro definition: "), (u'PDDM-DEFINE foo()\nbody\nPDDM-DEFINE foo(a)\nmumble\n', 'Attempt to redefine macro: ')]
        for (idx, (input_str, expected_prefix)) in enumerate(test_list, 1):
            f = io.StringIO(input_str)
            try:
                result = pddm.MacroCollection(f)
                self.fail('Should throw exception, entry %d' % idx)
            except pddm.PDDMError as e:
                self.assertTrue(e.message.startswith(expected_prefix), 'Entry %d failed: %r' % (idx, e))

    def testParseBeginIssues(self):
        if False:
            for i in range(10):
                print('nop')
        test_list = [(u'PDDM-DEFINE\nmumble', 'Failed to parse macro definition: '), (u'PDDM-DEFINE  \nmumble', 'Failed to parse macro definition: '), (u'PDDM-DEFINE  foo\nmumble', 'Failed to parse macro definition: '), (u'PDDM-DEFINE foo(\nmumble', 'Failed to parse macro definition: '), (u'PDDM-DEFINE foo(a, b\nmumble', 'Failed to parse macro definition: '), (u'PDDM-DEFINE  (a, b)\nmumble', 'Failed to parse macro definition: '), (u'PDDM-DEFINE foo bar(a, b)\nmumble', 'Failed to parse macro definition: '), (u'PDDM-DEFINE foo(a, ,b)\nmumble', 'Empty arg name in macro definition: '), (u'PDDM-DEFINE foo(a,,b)\nmumble', 'Empty arg name in macro definition: '), (u'PDDM-DEFINE foo(a,b,a,c)\nmumble', 'Arg name "a" used more than once in macro definition: '), (u'PDDM-DEFINE foo(a b,c)\nmumble', 'Invalid arg name "a b" in macro definition: '), (u'PDDM-DEFINE foo(a.b,c)\nmumble', 'Invalid arg name "a.b" in macro definition: '), (u'PDDM-DEFINE foo(a-b,c)\nmumble', 'Invalid arg name "a-b" in macro definition: '), (u'PDDM-DEFINE foo(a,b,c.)\nmumble', 'Invalid arg name "c." in macro definition: '), (u'PDDM-DEFINE foo(a,c) foo\nmumble', 'Failed to parse macro definition: '), (u'PDDM-DEFINE foo(a,c) foo)\nmumble', 'Failed to parse macro definition: ')]
        for (idx, (input_str, expected_prefix)) in enumerate(test_list, 1):
            f = io.StringIO(input_str)
            try:
                result = pddm.MacroCollection(f)
                self.fail('Should throw exception, entry %d' % idx)
            except pddm.PDDMError as e:
                self.assertTrue(e.message.startswith(expected_prefix), 'Entry %d failed: %r' % (idx, e))

class TestExpandingMacros(unittest.TestCase):

    def testExpandBasics(self):
        if False:
            return 10
        f = io.StringIO(u'\nPDDM-DEFINE noArgs( )\nbody1\nbody2\n\nPDDM-DEFINE-END\n\nPDDM-DEFINE oneArg(a)\nbody3 a\n\nPDDM-DEFINE-END\n\nPDDM-DEFINE twoArgs(b,c)\nbody4 b c\nbody5\nPDDM-DEFINE-END\n\n')
        mc = pddm.MacroCollection(f)
        test_list = [(u'noArgs()', 'body1\nbody2\n'), (u'oneArg(wee)', 'body3 wee\n'), (u'twoArgs(having some, fun)', 'body4 having some fun\nbody5'), (u'oneArg()', 'body3 \n'), (u'twoArgs(, empty)', 'body4  empty\nbody5'), (u'twoArgs(empty, )', 'body4 empty \nbody5'), (u'twoArgs(, )', 'body4  \nbody5')]
        for (idx, (input_str, expected)) in enumerate(test_list, 1):
            result = mc.Expand(input_str)
            self.assertEqual(result, expected, 'Entry %d --\n       Result: %r\n     Expected: %r' % (idx, result, expected))

    def testExpandArgOptions(self):
        if False:
            print('Hello World!')
        f = io.StringIO(u'\nPDDM-DEFINE bar(a)\na-a$S-a$l-a$L-a$u-a$U\nPDDM-DEFINE-END\n')
        mc = pddm.MacroCollection(f)
        self.assertEqual(mc.Expand('bar(xYz)'), 'xYz-   -xYz-xyz-XYz-XYZ')
        self.assertEqual(mc.Expand('bar(MnoP)'), 'MnoP-    -mnoP-mnop-MnoP-MNOP')
        self.assertEqual(mc.Expand('bar()'), '-----')

    def testExpandSimpleMacroErrors(self):
        if False:
            print('Hello World!')
        f = io.StringIO(u'\nPDDM-DEFINE foo(a, b)\n<a-z>\nPDDM-DEFINE baz(a)\na - a$z\n')
        mc = pddm.MacroCollection(f)
        test_list = [(u'bar()', 'No macro named "bar".'), (u'bar(a)', 'No macro named "bar".'), (u'foo()', 'Expected 2 args, got: "foo()".'), (u'foo(a b)', 'Expected 2 args, got: "foo(a b)".'), (u'foo(a,b,c)', 'Expected 2 args, got: "foo(a,b,c)".'), (u'baz(mumble)', 'Unknown arg option "a$z" while expanding "baz(mumble)".')]
        for (idx, (input_str, expected_err)) in enumerate(test_list, 1):
            try:
                result = mc.Expand(input_str)
                self.fail('Should throw exception, entry %d' % idx)
            except pddm.PDDMError as e:
                self.assertEqual(e.message, expected_err, 'Entry %d failed: %r' % (idx, e))

    def testExpandReferences(self):
        if False:
            print('Hello World!')
        f = io.StringIO(u'\nPDDM-DEFINE StartIt()\nfoo(abc, def)\nfoo(ghi, jkl)\nPDDM-DEFINE foo(a, b)\nbar(a, int)\nbar(b, NSString *)\nPDDM-DEFINE bar(n, t)\n- (t)n;\n- (void)set##n$u##:(t)value;\n\n')
        mc = pddm.MacroCollection(f)
        expected = '- (int)abc;\n- (void)setAbc:(int)value;\n\n- (NSString *)def;\n- (void)setDef:(NSString *)value;\n\n- (int)ghi;\n- (void)setGhi:(int)value;\n\n- (NSString *)jkl;\n- (void)setJkl:(NSString *)value;\n'
        self.assertEqual(mc.Expand('StartIt()'), expected)

    def testCatchRecursion(self):
        if False:
            print('Hello World!')
        f = io.StringIO(u'\nPDDM-DEFINE foo(a, b)\nbar(1, a)\nbar(2, b)\nPDDM-DEFINE bar(x, y)\nfoo(x, y)\n')
        mc = pddm.MacroCollection(f)
        try:
            result = mc.Expand('foo(A,B)')
            self.fail('Should throw exception, entry %d' % idx)
        except pddm.PDDMError as e:
            self.assertEqual(e.message, 'Found macro recusion, invoking "foo(1, A)":\n...while expanding "bar(1, A)".\n...while expanding "foo(A,B)".')

class TestParsingSource(unittest.TestCase):

    def testBasicParse(self):
        if False:
            i = 10
            return i + 15
        test_list = [(u'a\nb\nc', (3,)), (u'a\n//%PDDM-DEFINE foo()\n//%body\nc', (1, 2, 1)), (u'a\n//%PDDM-DEFINE foo()\n//%body\n//%PDDM-DEFINE bar()\n//%body2\nc', (1, 4, 1)), (u'a\n//%PDDM-DEFINE foo()\n//%body\n//%PDDM-DEFINE-END\n//%PDDM-DEFINE bar()\n//%body2\n//%PDDM-DEFINE-END\nc', (1, 6, 1)), (u'a\n//%PDDM-EXPAND foo()\nbody\n//%PDDM-EXPAND-END\n//%PDDM-DEFINE bar()\n//%body2\n', (1, 1, 2)), (u'a\nb\n//%PDDM-DEFINE bar()\n//%body2\n//%PDDM-EXPAND bar()\nbody2\n//%PDDM-EXPAND-END\n', (2, 2, 1)), (u'a\n//%PDDM-EXPAND foo(1)\nbody\n//%PDDM-EXPAND foo(2)\nbody2\n//%PDDM-EXPAND-END\n//%PDDM-DEFINE foo()\n//%body2\n', (1, 2, 2))]
        for (idx, (input_str, line_counts)) in enumerate(test_list, 1):
            f = io.StringIO(input_str)
            sf = pddm.SourceFile(f)
            sf._ParseFile()
            self.assertEqual(len(sf._sections), len(line_counts), 'Entry %d -- %d != %d' % (idx, len(sf._sections), len(line_counts)))
            for (idx2, (sec, expected)) in enumerate(zip(sf._sections, line_counts), 1):
                self.assertEqual(sec.num_lines_captured, expected, 'Entry %d, section %d -- %d != %d' % (idx, idx2, sec.num_lines_captured, expected))

    def testErrors(self):
        if False:
            print('Hello World!')
        test_list = [(u'//%PDDM-EXPAND a()\n//%PDDM-BOGUS', 'Ran into directive ("//%PDDM-BOGUS", line 2) while in "//%PDDM-EXPAND a()".'), (u'//%PDDM-EXPAND a()\n//%PDDM-DEFINE a()\n//%body\n', 'Ran into directive ("//%PDDM-DEFINE", line 2) while in "//%PDDM-EXPAND a()".'), (u'//%PDDM-EXPAND a()\na\nb\n', 'Hit the end of the file while in "//%PDDM-EXPAND a()".'), (u'//%PDDM-DEFINE a()\n//%body\n//%PDDM-BOGUS', 'Ran into directive ("//%PDDM-BOGUS", line 3) while in "//%PDDM-DEFINE a()".'), (u'//%PDDM-DEFINE a()\n//%body\n//%PDDM-EXPAND-END a()', 'Ran into directive ("//%PDDM-EXPAND-END", line 3) while in "//%PDDM-DEFINE a()".'), (u'a\n//%PDDM-DEFINE-END a()\n//a\n', 'Unexpected line 2: "//%PDDM-DEFINE-END a()".'), (u'a\n//%PDDM-EXPAND-END a()\n//a\n', 'Unexpected line 2: "//%PDDM-EXPAND-END a()".'), (u'//%PDDM-BOGUS\n//a\n', 'Unexpected line 1: "//%PDDM-BOGUS".')]
        for (idx, (input_str, expected_err)) in enumerate(test_list, 1):
            f = io.StringIO(input_str)
            try:
                pddm.SourceFile(f)._ParseFile()
                self.fail('Should throw exception, entry %d' % idx)
            except pddm.PDDMError as e:
                self.assertEqual(e.message, expected_err, 'Entry %d failed: %r' % (idx, e))

class TestProcessingSource(unittest.TestCase):

    def testBasics(self):
        if False:
            i = 10
            return i + 15
        input_str = u'\n//%PDDM-IMPORT-DEFINES ImportFile\nfoo\n//%PDDM-EXPAND mumble(abc)\n//%PDDM-EXPAND-END\nbar\n//%PDDM-EXPAND mumble(def)\n//%PDDM-EXPAND mumble(ghi)\n//%PDDM-EXPAND-END\nbaz\n//%PDDM-DEFINE mumble(a_)\n//%a_: getName(a_)\n'
        input_str2 = u'\n//%PDDM-DEFINE getName(x_)\n//%do##x_$u##(int x_);\n\n'
        expected = u'\n//%PDDM-IMPORT-DEFINES ImportFile\nfoo\n//%PDDM-EXPAND mumble(abc)\n// This block of code is generated, do not edit it directly.\n\nabc: doAbc(int abc);\n//%PDDM-EXPAND-END mumble(abc)\nbar\n//%PDDM-EXPAND mumble(def)\n// This block of code is generated, do not edit it directly.\n\ndef: doDef(int def);\n//%PDDM-EXPAND mumble(ghi)\n// This block of code is generated, do not edit it directly.\n\nghi: doGhi(int ghi);\n//%PDDM-EXPAND-END (2 expansions)\nbaz\n//%PDDM-DEFINE mumble(a_)\n//%a_: getName(a_)\n'
        expected_stripped = u'\n//%PDDM-IMPORT-DEFINES ImportFile\nfoo\n//%PDDM-EXPAND mumble(abc)\n//%PDDM-EXPAND-END mumble(abc)\nbar\n//%PDDM-EXPAND mumble(def)\n//%PDDM-EXPAND mumble(ghi)\n//%PDDM-EXPAND-END (2 expansions)\nbaz\n//%PDDM-DEFINE mumble(a_)\n//%a_: getName(a_)\n'

        def _Resolver(name):
            if False:
                i = 10
                return i + 15
            self.assertEqual(name, 'ImportFile')
            return io.StringIO(input_str2)
        f = io.StringIO(input_str)
        sf = pddm.SourceFile(f, _Resolver)
        sf.ProcessContent()
        self.assertEqual(sf.processed_content, expected)
        f2 = io.StringIO(sf.processed_content)
        sf2 = pddm.SourceFile(f2, _Resolver)
        sf2.ProcessContent()
        self.assertEqual(sf2.processed_content, expected)
        self.assertEqual(sf2.processed_content, sf.processed_content)
        f2 = io.StringIO(input_str)
        sf2 = pddm.SourceFile(f2)
        sf2.ProcessContent(strip_expansion=True)
        self.assertEqual(sf2.processed_content, expected_stripped)
        f2 = io.StringIO(sf.processed_content)
        sf2 = pddm.SourceFile(f2, _Resolver)
        sf2.ProcessContent(strip_expansion=True)
        self.assertEqual(sf2.processed_content, expected_stripped)

    def testProcessFileWithMacroParseError(self):
        if False:
            for i in range(10):
                print('nop')
        input_str = u'\nfoo\n//%PDDM-DEFINE mumble(a_)\n//%body\n//%PDDM-DEFINE mumble(x_)\n//%body2\n\n'
        f = io.StringIO(input_str)
        sf = pddm.SourceFile(f)
        try:
            sf.ProcessContent()
            self.fail('Should throw exception, entry %d' % idx)
        except pddm.PDDMError as e:
            self.assertEqual(e.message, 'Attempt to redefine macro: "PDDM-DEFINE mumble(x_)"\n...while parsing section that started:\n  Line 3: //%PDDM-DEFINE mumble(a_)')

    def testProcessFileWithExpandError(self):
        if False:
            while True:
                i = 10
        input_str = u'\nfoo\n//%PDDM-DEFINE mumble(a_)\n//%body\n//%PDDM-EXPAND foobar(x_)\n//%PDDM-EXPAND-END\n\n'
        f = io.StringIO(input_str)
        sf = pddm.SourceFile(f)
        try:
            sf.ProcessContent()
            self.fail('Should throw exception, entry %d' % idx)
        except pddm.PDDMError as e:
            self.assertEqual(e.message, 'No macro named "foobar".\n...while expanding "foobar(x_)" from the section that started:\n   Line 5: //%PDDM-EXPAND foobar(x_)')
if __name__ == '__main__':
    unittest.main()