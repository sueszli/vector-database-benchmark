import os
from pathlib import Path
import re
import subprocess
import sys
import matplotlib.pyplot as plt
from matplotlib.texmanager import TexManager
from matplotlib.testing._markers import needs_usetex
import pytest

def test_fontconfig_preamble():
    if False:
        while True:
            i = 10
    'Test that the preamble is included in the source.'
    plt.rcParams['text.usetex'] = True
    src1 = TexManager()._get_tex_source('', fontsize=12)
    plt.rcParams['text.latex.preamble'] = '\\usepackage{txfonts}'
    src2 = TexManager()._get_tex_source('', fontsize=12)
    assert src1 != src2

@pytest.mark.parametrize('rc, preamble, family', [({'font.family': 'sans-serif', 'font.sans-serif': 'helvetica'}, '\\usepackage{helvet}', '\\sffamily'), ({'font.family': 'serif', 'font.serif': 'palatino'}, '\\usepackage{mathpazo}', '\\rmfamily'), ({'font.family': 'cursive', 'font.cursive': 'zapf chancery'}, '\\usepackage{chancery}', '\\rmfamily'), ({'font.family': 'monospace', 'font.monospace': 'courier'}, '\\usepackage{courier}', '\\ttfamily'), ({'font.family': 'helvetica'}, '\\usepackage{helvet}', '\\sffamily'), ({'font.family': 'palatino'}, '\\usepackage{mathpazo}', '\\rmfamily'), ({'font.family': 'zapf chancery'}, '\\usepackage{chancery}', '\\rmfamily'), ({'font.family': 'courier'}, '\\usepackage{courier}', '\\ttfamily')])
def test_font_selection(rc, preamble, family):
    if False:
        for i in range(10):
            print('nop')
    plt.rcParams.update(rc)
    tm = TexManager()
    src = Path(tm.make_tex('hello, world', fontsize=12)).read_text()
    assert preamble in src
    assert [*re.findall('\\\\\\w+family', src)] == [family]

@needs_usetex
def test_unicode_characters():
    if False:
        for i in range(10):
            print('nop')
    plt.rcParams['text.usetex'] = True
    (fig, ax) = plt.subplots()
    ax.set_ylabel('\\textit{Velocity (°/sec)}')
    ax.set_xlabel('¼Öøæ')
    fig.canvas.draw()
    with pytest.raises(RuntimeError):
        ax.set_title('☃')
        fig.canvas.draw()

@needs_usetex
def test_openin_any_paranoid():
    if False:
        print('Hello World!')
    completed = subprocess.run([sys.executable, '-c', 'import matplotlib.pyplot as plt;plt.rcParams.update({"text.usetex": True});plt.title("paranoid");plt.show(block=False);'], env={**os.environ, 'openin_any': 'p'}, check=True, capture_output=True)
    assert completed.stderr == b''