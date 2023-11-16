"""Run doctests for tensorflow."""
import doctest
import re
import textwrap
import numpy as np

class _FloatExtractor(object):
    """Class for extracting floats from a string.

  For example:

  >>> text_parts, floats = _FloatExtractor()("Text 1.0 Text")
  >>> text_parts
  ["Text ", " Text"]
  >>> floats
  np.array([1.0])
  """
    _FLOAT_RE = re.compile('\n      (                          # Captures the float value.\n        (?:\n           [-+]|                 # Start with a sign is okay anywhere.\n           (?:                   # Otherwise:\n               ^|                # Start after the start of string\n               (?<=[^\\w.])       # Not after a word char, or a .\n           )\n        )\n        (?:                      # Digits and exponent - something like:\n          {digits_dot_maybe_digits}{exponent}?|   # "1.0" "1." "1.0e3", "1.e3"\n          {dot_digits}{exponent}?|                # ".1" ".1e3"\n          {digits}{exponent}|                     # "1e3"\n          {digits}(?=j)                           # "300j"\n        )\n      )\n      j?                         # Optional j for cplx numbers, not captured.\n      (?=                        # Only accept the match if\n        $|                       # * At the end of the string, or\n        [^\\w.]                   # * Next char is not a word char or "."\n      )\n      '.format(digits_dot_maybe_digits='(?:[0-9]+\\.(?:[0-9]*))', dot_digits='(?:\\.[0-9]+)', digits='(?:[0-9]+)', exponent='(?:[eE][-+]?[0-9]+)'), re.VERBOSE)

    def __call__(self, string):
        if False:
            while True:
                i = 10
        'Extracts floats from a string.\n\n    >>> text_parts, floats = _FloatExtractor()("Text 1.0 Text")\n    >>> text_parts\n    ["Text ", " Text"]\n    >>> floats\n    np.array([1.0])\n\n    Args:\n      string: the string to extract floats from.\n\n    Returns:\n      A (string, array) pair, where `string` has each float replaced by "..."\n      and `array` is a `float32` `numpy.array` containing the extracted floats.\n    '
        texts = []
        floats = []
        for (i, part) in enumerate(self._FLOAT_RE.split(string)):
            if i % 2 == 0:
                texts.append(part)
            else:
                floats.append(float(part))
        return (texts, np.array(floats))

class TfDoctestOutputChecker(doctest.OutputChecker, object):
    """Customizes how `want` and `got` are compared, see `check_output`."""

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(TfDoctestOutputChecker, self).__init__(*args, **kwargs)
        self.extract_floats = _FloatExtractor()
        self.text_good = None
        self.float_size_good = None
    _ADDRESS_RE = re.compile('\\bat 0x[0-9a-f]*?>')
    _NUMPY_OUTPUT_RE = re.compile('<tf.Tensor.*?numpy=(.*?)>', re.DOTALL)

    def _allclose(self, want, got, rtol=0.001, atol=0.001):
        if False:
            print('Hello World!')
        return np.allclose(want, got, rtol=rtol, atol=atol)

    def _tf_tensor_numpy_output(self, string):
        if False:
            print('Hello World!')
        modified_string = self._NUMPY_OUTPUT_RE.sub('\\1', string)
        return (modified_string, modified_string != string)
    MESSAGE = textwrap.dedent('\n\n        #############################################################\n        Check the documentation (https://www.tensorflow.org/community/contribute/docs_ref) on how to\n        write testable docstrings.\n        #############################################################')

    def check_output(self, want, got, optionflags):
        if False:
            return 10
        'Compares the docstring output to the output gotten by running the code.\n\n    Python addresses in the output are replaced with wildcards.\n\n    Float values in the output compared as using `np.allclose`:\n\n      * Float values are extracted from the text and replaced with wildcards.\n      * The wildcard text is compared to the actual output.\n      * The float values are compared using `np.allclose`.\n\n    The method returns `True` if both the text comparison and the numeric\n    comparison are successful.\n\n    The numeric comparison will fail if either:\n\n      * The wrong number of floats are found.\n      * The float values are not within tolerence.\n\n    Args:\n      want: The output in the docstring.\n      got: The output generated after running the snippet.\n      optionflags: Flags passed to the doctest.\n\n    Returns:\n      A bool, indicating if the check was successful or not.\n    '
        if got and (not want):
            return True
        if want is None:
            want = ''
        if want == got:
            return True
        want = self._ADDRESS_RE.sub('at ...>', want)
        (want, want_changed) = self._tf_tensor_numpy_output(want)
        if want_changed:
            (got, _) = self._tf_tensor_numpy_output(got)
        (want_text_parts, self.want_floats) = self.extract_floats(want)
        want_text_parts = [part.strip(' ') for part in want_text_parts]
        want_text_wild = '...'.join(want_text_parts)
        if '....' in want_text_wild:
            want_text_wild = re.sub('\\.\\.\\.\\.+', '...', want_text_wild)
        (_, self.got_floats) = self.extract_floats(got)
        self.text_good = super(TfDoctestOutputChecker, self).check_output(want=want_text_wild, got=got, optionflags=optionflags)
        if not self.text_good:
            return False
        if self.want_floats.size == 0:
            return True
        self.float_size_good = self.want_floats.size == self.got_floats.size
        if self.float_size_good:
            return self._allclose(self.want_floats, self.got_floats)
        else:
            return False

    def output_difference(self, example, got, optionflags):
        if False:
            while True:
                i = 10
        got = [got]
        if self.text_good:
            if not self.float_size_good:
                got.append('\n\nCAUTION: tf_doctest doesn\'t work if *some* of the *float output* is hidden with a "...".')
        got.append(self.MESSAGE)
        got = '\n'.join(got)
        return super(TfDoctestOutputChecker, self).output_difference(example, got, optionflags)