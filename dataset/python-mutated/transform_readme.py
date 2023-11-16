"""Transform the README.md to support a specific deployment target.

By default, we assume that our README.md will be rendered on GitHub. However, different
targets have different strategies for rendering light- and dark-mode images. This script
adjusts the images in the README.md to support the given target.
"""
from __future__ import annotations
import argparse
from pathlib import Path
URL = 'https://user-images.githubusercontent.com/1309177/{}.svg'
URL_LIGHT = URL.format('232603516-4fb4892d-585c-4b20-b810-3db9161831e4')
URL_DARK = URL.format('232603514-c95e9b0f-6b31-43de-9a80-9e844173fd6a')
GITHUB = f'\n<p align="center">\n  <picture align="center">\n    <source media="(prefers-color-scheme: dark)" srcset="{URL_DARK}">\n    <source media="(prefers-color-scheme: light)" srcset="{URL_LIGHT}">\n    <img alt="Shows a bar chart with benchmark results." src="{URL_LIGHT}">\n  </picture>\n</p>\n'
PYPI = f'\n<p align="center">\n  <img alt="Shows a bar chart with benchmark results." src="{URL_LIGHT}">\n</p>\n'
MK_DOCS = f'\n<p align="center">\n  <img alt="Shows a bar chart with benchmark results." src="{URL_LIGHT}#only-light">\n</p>\n\n<p align="center">\n  <img alt="Shows a bar chart with benchmark results." src="{URL_DARK}#only-dark">\n</p>\n'

def main(target: str) -> None:
    if False:
        return 10
    'Modify the README.md to support the given target.'
    with Path('README.md').open(encoding='utf8') as fp:
        content = fp.read()
        if GITHUB not in content:
            msg = 'README.md is not in the expected format.'
            raise ValueError(msg)
    if target == 'pypi':
        with Path('README.md').open('w', encoding='utf8') as fp:
            fp.write(content.replace(GITHUB, PYPI))
    elif target == 'mkdocs':
        with Path('README.md').open('w', encoding='utf8') as fp:
            fp.write(content.replace(GITHUB, MK_DOCS))
    else:
        msg = f'Unknown target: {target}'
        raise ValueError(msg)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modify the README.md to support a specific deployment target.')
    parser.add_argument('--target', type=str, required=True, choices=('pypi', 'mkdocs'))
    args = parser.parse_args()
    main(target=args.target)