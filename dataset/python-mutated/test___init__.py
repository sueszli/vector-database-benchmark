from __future__ import annotations
import pytest
pytest
import re
import warnings
from bokeh.util.warnings import BokehDeprecationWarning, BokehUserWarning
from tests.support.util.api import verify_all
from tests.support.util.types import Capture
import bokeh as b
ALL = ('__version__', 'license', 'sampledata')
_LICENSE = 'Copyright (c) 2012 - 2023, Anaconda, Inc., and Bokeh Contributors\nAll rights reserved.\n\nRedistribution and use in source and binary forms, with or without modification,\nare permitted provided that the following conditions are met:\n\nRedistributions of source code must retain the above copyright notice,\nthis list of conditions and the following disclaimer.\n\nRedistributions in binary form must reproduce the above copyright notice,\nthis list of conditions and the following disclaimer in the documentation\nand/or other materials provided with the distribution.\n\nNeither the name of Anaconda nor the names of any contributors\nmay be used to endorse or promote products derived from this software\nwithout specific prior written permission.\n\nTHIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"\nAND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE\nIMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE\nARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE\nLIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR\nCONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF\nSUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS\nINTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN\nCONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)\nARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF\nTHE POSSIBILITY OF SUCH DAMAGE.\n\n'
Test___all__ = verify_all(b, ALL)

def test___version___type() -> None:
    if False:
        while True:
            i = 10
    assert isinstance(b.__version__, str)

def test___version___defined() -> None:
    if False:
        return 10
    VERSION_PAT = re.compile('^(\\d+\\.\\d+\\.\\d+)((?:\\.dev|\\.rc).*)?')
    assert VERSION_PAT.match(b.__version__.strip('.dirty'))

def test_license(capsys: Capture) -> None:
    if False:
        for i in range(10):
            print('nop')
    b.license()
    (out, err) = capsys.readouterr()
    assert out == _LICENSE

class TestWarnings:

    @pytest.mark.parametrize('cat', (BokehDeprecationWarning, BokehUserWarning))
    def test_bokeh_custom(self, cat) -> None:
        if False:
            return 10
        r = warnings.formatwarning('message', cat, 'line', 'lineno')
        assert r == f'{cat.__name__}: message\n'

    def test_general_default(self) -> None:
        if False:
            i = 10
            return i + 15
        r = warnings.formatwarning('message', RuntimeWarning, 'line', 'lineno')
        assert r == 'line:lineno: RuntimeWarning: message\n'

    @pytest.mark.skip
    def test_filters(self) -> None:
        if False:
            while True:
                i = 10
        assert ('always', None, BokehUserWarning, None, 0) in warnings.filters
        assert ('always', None, BokehDeprecationWarning, None, 0) in warnings.filters