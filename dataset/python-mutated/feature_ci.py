from hatch.template import File
from hatch.utils.fs import Path
from .default import get_files as get_template_files

def get_files(**kwargs):
    if False:
        while True:
            i = 10
    files = [File(Path(f.path), f.contents) for f in get_template_files(**kwargs)]
    files.append(File(Path('.github', 'workflows', 'test.yml'), 'name: test\n\non:\n  push:\n    branches: [main, master]\n  pull_request:\n    branches: [main, master]\n\nconcurrency:\n  group: test-${{ github.head_ref }}\n  cancel-in-progress: true\n\nenv:\n  PYTHONUNBUFFERED: "1"\n  FORCE_COLOR: "1"\n\njobs:\n  run:\n    name: Python ${{ matrix.python-version }} on ${{ startsWith(matrix.os, \'macos-\') && \'macOS\' || startsWith(matrix.os, \'windows-\') && \'Windows\' || \'Linux\' }}\n    runs-on: ${{ matrix.os }}\n    strategy:\n      fail-fast: false\n      matrix:\n        os: [ubuntu-latest, windows-latest, macos-latest]\n        python-version: [\'3.8\', \'3.9\', \'3.10\', \'3.11\', \'3.12\']\n\n    steps:\n    - uses: actions/checkout@v3\n\n    - name: Set up Python ${{ matrix.python-version }}\n      uses: actions/setup-python@v4\n      with:\n        python-version: ${{ matrix.python-version }}\n\n    - name: Install Hatch\n      run: pip install --upgrade hatch\n\n    - name: Run tests\n      run: hatch run cov\n'))
    return files