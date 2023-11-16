"""Tests for fenced_doctest."""
from typing import List, Optional, Tuple
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow.tools.docs import fenced_doctest_lib
EXAMPLES = [('simple', [('code', None)], '\n     Hello\n\n     ``` python\n     code\n     ```\n\n     Goodbye\n     '), ('output', [('code', 'result')], '\n     Hello\n\n     ``` python\n     code\n     ```\n\n     ```\n     result\n     ```\n\n     Goodbye\n     '), ('not-output', [('code', None)], '\n     Hello\n\n     ``` python\n     code\n     ```\n\n     ``` bash\n     result\n     ```\n\n     Goodbye\n     '), ('first', [('code', None)], '\n     ``` python\n     code\n     ```\n\n     Goodbye\n     '[1:]), ('last', [('code', None)], '\n     Hello\n\n     ``` python\n     code\n     ```'), ('last_output', [('code', 'result')], '\n     Hello\n\n     ``` python\n     code\n     ```\n\n     ```\n     result\n     ```'), ('skip-unlabeled', [], '\n     Hello\n\n     ```\n     skip\n     ```\n\n     Goodbye\n     '), ('skip-wrong-label', [], '\n     Hello\n\n     ``` sdkfjgsd\n     skip\n     ```\n\n     Goodbye\n     '), ('doctest_skip', [], '\n     Hello\n\n     ``` python\n     doctest: +SKIP\n     ```\n\n     Goodbye\n     '), ('skip_all', [], '\n     <!-- doctest: skip-all -->\n\n     Hello\n\n     ``` python\n     a\n     ```\n\n     ``` python\n     b\n     ```\n\n     Goodbye\n     '), ('two', [('a', None), ('b', None)], '\n     Hello\n\n     ``` python\n     a\n     ```\n\n     ``` python\n     b\n     ```\n\n     Goodbye\n     '), ('two-outputs', [('a', 'A'), ('b', 'B')], '\n     Hello\n\n     ``` python\n     a\n     ```\n\n     ```\n     A\n     ```\n\n     ``` python\n     b\n     ```\n\n     ```\n     B\n     ```\n\n     Goodbye\n     '), ('list', [('a', None), ('b', 'B'), ('c', 'C'), ('d', None)], '\n     Hello\n\n     ``` python\n     a\n     ```\n\n     ``` python\n     b\n     ```\n\n     ```\n     B\n     ```\n\n     List:\n     * first\n\n       ``` python\n       c\n       ```\n\n       ```\n       C\n       ```\n\n       ``` python\n       d\n       ```\n     * second\n\n\n     Goodbye\n     '), ('multiline', [('a\nb', 'A\nB')], '\n     Hello\n\n     ``` python\n     a\n     b\n     ```\n\n     ```\n     A\n     B\n     ```\n\n     Goodbye\n     ')]
ExampleTuples = List[Tuple[str, Optional[str]]]

class G3DoctestTest(parameterized.TestCase):

    def _do_test(self, expected_example_tuples, string):
        if False:
            for i in range(10):
                print('nop')
        parser = fenced_doctest_lib.FencedCellParser(fence_label='python')
        example_tuples = []
        for example in parser.get_examples(string, name=self._testMethodName):
            source = example.source.rstrip('\n')
            want = example.want
            if want is not None:
                want = want.rstrip('\n')
            example_tuples.append((source, want))
        self.assertEqual(expected_example_tuples, example_tuples)

    @parameterized.named_parameters(*EXAMPLES)
    def test_parser(self, expected_example_tuples: ExampleTuples, string: str):
        if False:
            return 10
        self._do_test(expected_example_tuples, string)

    @parameterized.named_parameters(*EXAMPLES)
    def test_parser_no_blanks(self, expected_example_tuples: ExampleTuples, string: str):
        if False:
            i = 10
            return i + 15
        string = string.replace('\n\n', '\n')
        self._do_test(expected_example_tuples, string)
if __name__ == '__main__':
    absltest.main()