# Copyright (c) 2021 elParaguayo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import sys
from importlib import reload

import pytest

from test.widgets.test_net import MockPsutil


@pytest.fixture
def widget(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", MockPsutil("psutil"))
    from libqtile.widget import net

    reload(net)
    yield net.Net


@pytest.mark.parametrize(
    "screenshot_manager",
    [
        {},
        {"format": "{interface}: U {up} D {down} T {total}"},
        {
            "format": "{interface}: U {up}{up_suffix} D {down}{down_suffix} T {total}{total_suffix}"
        },
        {"format": "{down:.0f}{down_suffix} \u2193\u2191 {up:.0f}{up_suffix}"},
        {"interface": "wlp58s0"},
        {"prefix": "M"},
    ],
    indirect=True,
)
def ss_net(screenshot_manager):
    screenshot_manager.take_screenshot()
