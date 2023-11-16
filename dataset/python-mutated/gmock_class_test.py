"""Tests for gmock.scripts.generator.cpp.gmock_class."""
__author__ = 'nnorwitz@google.com (Neal Norwitz)'
import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cpp import ast
from cpp import gmock_class

class TestCase(unittest.TestCase):
    """Helper class that adds assert methods."""

    def StripLeadingWhitespace(self, lines):
        if False:
            while True:
                i = 10
        "Strip leading whitespace in each line in 'lines'."
        return '\n'.join([s.lstrip() for s in lines.split('\n')])

    def assertEqualIgnoreLeadingWhitespace(self, expected_lines, lines):
        if False:
            for i in range(10):
                print('nop')
        'Specialized assert that ignores the indent level.'
        self.assertEqual(expected_lines, self.StripLeadingWhitespace(lines))

class GenerateMethodsTest(TestCase):

    def GenerateMethodSource(self, cpp_source):
        if False:
            for i in range(10):
                print('nop')
        'Convert C++ source to Google Mock output source lines.'
        method_source_lines = []
        builder = ast.BuilderFromSource(cpp_source, '<test>')
        ast_list = list(builder.Generate())
        gmock_class._GenerateMethods(method_source_lines, cpp_source, ast_list[0])
        return '\n'.join(method_source_lines)

    def testSimpleMethod(self):
        if False:
            print('Hello World!')
        source = '\nclass Foo {\n public:\n  virtual int Bar();\n};\n'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD0(Bar,\nint());', self.GenerateMethodSource(source))

    def testSimpleConstructorsAndDestructor(self):
        if False:
            while True:
                i = 10
        source = '\nclass Foo {\n public:\n  Foo();\n  Foo(int x);\n  Foo(const Foo& f);\n  Foo(Foo&& f);\n  ~Foo();\n  virtual int Bar() = 0;\n};\n'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD0(Bar,\nint());', self.GenerateMethodSource(source))

    def testVirtualDestructor(self):
        if False:
            while True:
                i = 10
        source = '\nclass Foo {\n public:\n  virtual ~Foo();\n  virtual int Bar() = 0;\n};\n'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD0(Bar,\nint());', self.GenerateMethodSource(source))

    def testExplicitlyDefaultedConstructorsAndDestructor(self):
        if False:
            print('Hello World!')
        source = '\nclass Foo {\n public:\n  Foo() = default;\n  Foo(const Foo& f) = default;\n  Foo(Foo&& f) = default;\n  ~Foo() = default;\n  virtual int Bar() = 0;\n};\n'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD0(Bar,\nint());', self.GenerateMethodSource(source))

    def testExplicitlyDeletedConstructorsAndDestructor(self):
        if False:
            while True:
                i = 10
        source = '\nclass Foo {\n public:\n  Foo() = delete;\n  Foo(const Foo& f) = delete;\n  Foo(Foo&& f) = delete;\n  ~Foo() = delete;\n  virtual int Bar() = 0;\n};\n'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD0(Bar,\nint());', self.GenerateMethodSource(source))

    def testSimpleOverrideMethod(self):
        if False:
            return 10
        source = '\nclass Foo {\n public:\n  int Bar() override;\n};\n'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD0(Bar,\nint());', self.GenerateMethodSource(source))

    def testSimpleConstMethod(self):
        if False:
            while True:
                i = 10
        source = '\nclass Foo {\n public:\n  virtual void Bar(bool flag) const;\n};\n'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_CONST_METHOD1(Bar,\nvoid(bool flag));', self.GenerateMethodSource(source))

    def testExplicitVoid(self):
        if False:
            i = 10
            return i + 15
        source = '\nclass Foo {\n public:\n  virtual int Bar(void);\n};\n'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD0(Bar,\nint(void));', self.GenerateMethodSource(source))

    def testStrangeNewlineInParameter(self):
        if False:
            for i in range(10):
                print('nop')
        source = '\nclass Foo {\n public:\n  virtual void Bar(int\na) = 0;\n};\n'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD1(Bar,\nvoid(int a));', self.GenerateMethodSource(source))

    def testDefaultParameters(self):
        if False:
            while True:
                i = 10
        source = "\nclass Foo {\n public:\n  virtual void Bar(int a, char c = 'x') = 0;\n};\n"
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD2(Bar,\nvoid(int, char));', self.GenerateMethodSource(source))

    def testMultipleDefaultParameters(self):
        if False:
            print('Hello World!')
        source = "\nclass Foo {\n public:\n  virtual void Bar(int a = 42, char c = 'x') = 0;\n};\n"
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD2(Bar,\nvoid(int, char));', self.GenerateMethodSource(source))

    def testRemovesCommentsWhenDefaultsArePresent(self):
        if False:
            print('Hello World!')
        source = "\nclass Foo {\n public:\n  virtual void Bar(int a = 42 /* a comment */,\n                   char /* other comment */ c= 'x') = 0;\n};\n"
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD2(Bar,\nvoid(int, char));', self.GenerateMethodSource(source))

    def testDoubleSlashCommentsInParameterListAreRemoved(self):
        if False:
            return 10
        source = '\nclass Foo {\n public:\n  virtual void Bar(int a,  // inline comments should be elided.\n                   int b   // inline comments should be elided.\n                   ) const = 0;\n};\n'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_CONST_METHOD2(Bar,\nvoid(int a, int b));', self.GenerateMethodSource(source))

    def testCStyleCommentsInParameterListAreNotRemoved(self):
        if False:
            print('Hello World!')
        source = '\nclass Foo {\n public:\n  virtual const string& Bar(int /* keeper */, int b);\n};\n'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD2(Bar,\nconst string&(int /* keeper */, int b));', self.GenerateMethodSource(source))

    def testArgsOfTemplateTypes(self):
        if False:
            while True:
                i = 10
        source = '\nclass Foo {\n public:\n  virtual int Bar(const vector<int>& v, map<int, string>* output);\n};'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD2(Bar,\nint(const vector<int>& v, map<int, string>* output));', self.GenerateMethodSource(source))

    def testReturnTypeWithOneTemplateArg(self):
        if False:
            for i in range(10):
                print('nop')
        source = '\nclass Foo {\n public:\n  virtual vector<int>* Bar(int n);\n};'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD1(Bar,\nvector<int>*(int n));', self.GenerateMethodSource(source))

    def testReturnTypeWithManyTemplateArgs(self):
        if False:
            print('Hello World!')
        source = '\nclass Foo {\n public:\n  virtual map<int, string> Bar();\n};'
        self.assertEqualIgnoreLeadingWhitespace("// The following line won't really compile, as the return\n// type has multiple template arguments.  To fix it, use a\n// typedef for the return type.\nMOCK_METHOD0(Bar,\nmap<int, string>());", self.GenerateMethodSource(source))

    def testSimpleMethodInTemplatedClass(self):
        if False:
            while True:
                i = 10
        source = '\ntemplate<class T>\nclass Foo {\n public:\n  virtual int Bar();\n};\n'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD0_T(Bar,\nint());', self.GenerateMethodSource(source))

    def testPointerArgWithoutNames(self):
        if False:
            for i in range(10):
                print('nop')
        source = '\nclass Foo {\n  virtual int Bar(C*);\n};\n'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD1(Bar,\nint(C*));', self.GenerateMethodSource(source))

    def testReferenceArgWithoutNames(self):
        if False:
            return 10
        source = '\nclass Foo {\n  virtual int Bar(C&);\n};\n'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD1(Bar,\nint(C&));', self.GenerateMethodSource(source))

    def testArrayArgWithoutNames(self):
        if False:
            print('Hello World!')
        source = '\nclass Foo {\n  virtual int Bar(C[]);\n};\n'
        self.assertEqualIgnoreLeadingWhitespace('MOCK_METHOD1(Bar,\nint(C[]));', self.GenerateMethodSource(source))

class GenerateMocksTest(TestCase):

    def GenerateMocks(self, cpp_source):
        if False:
            return 10
        'Convert C++ source to complete Google Mock output source.'
        filename = '<test>'
        builder = ast.BuilderFromSource(cpp_source, filename)
        ast_list = list(builder.Generate())
        lines = gmock_class._GenerateMocks(filename, cpp_source, ast_list, None)
        return '\n'.join(lines)

    def testNamespaces(self):
        if False:
            print('Hello World!')
        source = '\nnamespace Foo {\nnamespace Bar { class Forward; }\nnamespace Baz {\n\nclass Test {\n public:\n  virtual void Foo();\n};\n\n}  // namespace Baz\n}  // namespace Foo\n'
        expected = 'namespace Foo {\nnamespace Baz {\n\nclass MockTest : public Test {\npublic:\nMOCK_METHOD0(Foo,\nvoid());\n};\n\n}  // namespace Baz\n}  // namespace Foo\n'
        self.assertEqualIgnoreLeadingWhitespace(expected, self.GenerateMocks(source))

    def testClassWithStorageSpecifierMacro(self):
        if False:
            while True:
                i = 10
        source = '\nclass STORAGE_SPECIFIER Test {\n public:\n  virtual void Foo();\n};\n'
        expected = 'class MockTest : public Test {\npublic:\nMOCK_METHOD0(Foo,\nvoid());\n};\n'
        self.assertEqualIgnoreLeadingWhitespace(expected, self.GenerateMocks(source))

    def testTemplatedForwardDeclaration(self):
        if False:
            return 10
        source = '\ntemplate <class T> class Forward;  // Forward declaration should be ignored.\nclass Test {\n public:\n  virtual void Foo();\n};\n'
        expected = 'class MockTest : public Test {\npublic:\nMOCK_METHOD0(Foo,\nvoid());\n};\n'
        self.assertEqualIgnoreLeadingWhitespace(expected, self.GenerateMocks(source))

    def testTemplatedClass(self):
        if False:
            i = 10
            return i + 15
        source = '\ntemplate <typename S, typename T>\nclass Test {\n public:\n  virtual void Foo();\n};\n'
        expected = 'template <typename T0, typename T1>\nclass MockTest : public Test<T0, T1> {\npublic:\nMOCK_METHOD0_T(Foo,\nvoid());\n};\n'
        self.assertEqualIgnoreLeadingWhitespace(expected, self.GenerateMocks(source))

    def testTemplateInATemplateTypedef(self):
        if False:
            while True:
                i = 10
        source = '\nclass Test {\n public:\n  typedef std::vector<std::list<int>> FooType;\n  virtual void Bar(const FooType& test_arg);\n};\n'
        expected = 'class MockTest : public Test {\npublic:\nMOCK_METHOD1(Bar,\nvoid(const FooType& test_arg));\n};\n'
        self.assertEqualIgnoreLeadingWhitespace(expected, self.GenerateMocks(source))

    def testTemplateInATemplateTypedefWithComma(self):
        if False:
            i = 10
            return i + 15
        source = '\nclass Test {\n public:\n  typedef std::function<void(\n      const vector<std::list<int>>&, int> FooType;\n  virtual void Bar(const FooType& test_arg);\n};\n'
        expected = 'class MockTest : public Test {\npublic:\nMOCK_METHOD1(Bar,\nvoid(const FooType& test_arg));\n};\n'
        self.assertEqualIgnoreLeadingWhitespace(expected, self.GenerateMocks(source))

    def testEnumClass(self):
        if False:
            return 10
        source = '\nclass Test {\n public:\n  enum class Baz { BAZINGA };\n  virtual void Bar(const FooType& test_arg);\n};\n'
        expected = 'class MockTest : public Test {\npublic:\nMOCK_METHOD1(Bar,\nvoid(const FooType& test_arg));\n};\n'
        self.assertEqualIgnoreLeadingWhitespace(expected, self.GenerateMocks(source))
if __name__ == '__main__':
    unittest.main()