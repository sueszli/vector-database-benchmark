import io
import textwrap
import threading
import types
import os
import sys
from pysnooper.utils import truncate
import pytest
import pysnooper
from pysnooper.variables import needs_parentheses
from ..utils import assert_output, assert_sample_output, VariableEntry, CallEntry, LineEntry, ReturnEntry, OpcodeEntry, ReturnValueEntry, ExceptionEntry, ExceptionValueEntry, SourcePathEntry, CallEndedByExceptionEntry, ElapsedTimeEntry
from .. import mini_toolbox
from .multiple_files import foo

def test_multiple_files():
    if False:
        for i in range(10):
            print('nop')
    with mini_toolbox.OutputCapturer(stdout=False, stderr=True) as output_capturer:
        result = foo.foo_function()
    assert result == 21
    output = output_capturer.string_io.getvalue()
    assert_output(output, (SourcePathEntry(source_path_regex='.*foo\\.py$'), CallEntry(), LineEntry(), SourcePathEntry(source_path_regex='.*bar\\.py$'), VariableEntry(), CallEntry(), LineEntry(), VariableEntry(), LineEntry(), ReturnEntry(), ReturnValueEntry(), SourcePathEntry(source_path_regex='.*foo\\.py$'), VariableEntry(), LineEntry(), ReturnEntry(), ReturnValueEntry(), ElapsedTimeEntry()))