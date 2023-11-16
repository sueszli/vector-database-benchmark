"""Interface to pngquant executable."""
from __future__ import annotations
from pathlib import Path
from subprocess import PIPE
from packaging.version import Version
from ocrmypdf.exceptions import MissingDependencyError
from ocrmypdf.subprocess import get_version, run

def version() -> Version:
    if False:
        for i in range(10):
            print('nop')
    return Version(get_version('pngquant', regex='(\\d+(\\.\\d+)*).*'))

def available():
    if False:
        i = 10
        return i + 15
    try:
        version()
    except MissingDependencyError:
        return False
    return True

def quantize(input_file: Path, output_file: Path, quality_min: int, quality_max: int):
    if False:
        i = 10
        return i + 15
    'Quantize a PNG image using pngquant.\n\n    Args:\n        input_file: Input PNG image\n        output_file: Output PNG image\n        quality_min: Minimum quality to use\n        quality_max: Maximum quality to use\n    '
    with open(input_file, 'rb') as input_stream:
        args = ['pngquant', '--force', '--skip-if-larger', '--quality', f'{quality_min}-{quality_max}', '--', '-']
        result = run(args, stdin=input_stream, stdout=PIPE, stderr=PIPE, check=False)
    if result.returncode == 0:
        output_file.write_bytes(result.stdout)