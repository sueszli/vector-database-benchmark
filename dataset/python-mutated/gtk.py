from sympy.printing.mathml import mathml
from sympy.utilities.mathml import c2p
import tempfile
import subprocess

def print_gtk(x, start_viewer=True):
    if False:
        while True:
            i = 10
    'Print to Gtkmathview, a gtk widget capable of rendering MathML.\n\n    Needs libgtkmathview-bin'
    with tempfile.NamedTemporaryFile('w') as file:
        file.write(c2p(mathml(x), simple=True))
        file.flush()
        if start_viewer:
            subprocess.check_call(('mathmlviewer', file.name))