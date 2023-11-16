import os
import hypothesistooling as tools
from hypothesistooling.projects import hypothesispython as hp
from hypothesistooling.scripts import pip_tool

def is_sphinx(f):
    if False:
        print('Hello World!')
    f = os.path.abspath(f)
    return f.startswith(os.path.join(hp.HYPOTHESIS_PYTHON, 'docs'))
ALL_RST = [f for f in tools.all_files() if os.path.basename(f) != 'RELEASE.rst' and f.endswith('.rst')]

def test_passes_rst_lint():
    if False:
        for i in range(10):
            print('nop')
    pip_tool('rst-lint', *(f for f in ALL_RST if not is_sphinx(f)))

def disabled_test_passes_flake8():
    if False:
        print('Hello World!')
    pip_tool('flake8', '--select=W191,W291,W292,W293,W391', *ALL_RST)