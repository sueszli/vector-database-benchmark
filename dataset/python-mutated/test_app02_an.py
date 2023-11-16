import subprocess
import sys
from docs_src.testing.app02_an import main as mod
from docs_src.testing.app02_an.test_main import test_app

def test_app02_an():
    if False:
        i = 10
        return i + 15
    test_app()

def test_script():
    if False:
        i = 10
        return i + 15
    result = subprocess.run([sys.executable, '-m', 'coverage', 'run', mod.__file__, '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    assert 'Usage' in result.stdout