import importlib.metadata
import sys
from transformers.testing_utils import TestCasePlus
from transformers.utils.versions import require_version, require_version_core
numpy_ver = importlib.metadata.version('numpy')
python_ver = '.'.join([str(x) for x in sys.version_info[:3]])

class DependencyVersionCheckTest(TestCasePlus):

    def test_core(self):
        if False:
            return 10
        require_version_core('numpy<1000.4.5')
        require_version_core('numpy<1000.4')
        require_version_core('numpy<1000')
        require_version_core('numpy<=1000.4.5')
        require_version_core(f'numpy<={numpy_ver}')
        require_version_core(f'numpy=={numpy_ver}')
        require_version_core('numpy!=1000.4.5')
        require_version_core('numpy>=1.0')
        require_version_core('numpy>=1.0.0')
        require_version_core(f'numpy>={numpy_ver}')
        require_version_core('numpy>1.0.0')
        require_version_core('numpy>1.0.0,<1000')
        require_version_core('numpy')
        for req in ['numpy==1.0.0', 'numpy>=1000.0.0', f'numpy<{numpy_ver}']:
            try:
                require_version_core(req)
            except ImportError as e:
                self.assertIn(f'{req} is required', str(e))
                self.assertIn('but found', str(e))
        for req in ['numpipypie>1', 'numpipypie2']:
            try:
                require_version_core(req)
            except importlib.metadata.PackageNotFoundError as e:
                self.assertIn(f"The '{req}' distribution was not found and is required by this application", str(e))
                self.assertIn('Try: pip install transformers -U', str(e))
        for req in ['numpy??1.0.0', 'numpy1.0.0']:
            try:
                require_version_core(req)
            except ValueError as e:
                self.assertIn('requirement needs to be in the pip package format', str(e))
        for req in ['numpy=1.0.0', 'numpy == 1.00', 'numpy<>1.0.0', 'numpy><1.00', 'numpy>>1.0.0']:
            try:
                require_version_core(req)
            except ValueError as e:
                self.assertIn('need one of ', str(e))

    def test_python(self):
        if False:
            for i in range(10):
                print('nop')
        require_version('python>=3.6.0')
        for req in ['python>9.9.9', 'python<3.0.0']:
            try:
                require_version_core(req)
            except ImportError as e:
                self.assertIn(f'{req} is required', str(e))
                self.assertIn(f'but found python=={python_ver}', str(e))