from hatch.template import File
from hatch.utils.fs import Path
from hatchling.metadata.spec import DEFAULT_METADATA_VERSION
from ..new.feature_no_src_layout import get_files as get_template_files

def get_files(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    relative_root = kwargs.get('relative_root', '')
    files = []
    for f in get_template_files(**kwargs):
        part = f.path.parts[0]
        if part in ('my_app', 'pyproject.toml', 'README.md', 'LICENSE.txt'):
            files.append(File(Path(relative_root, f.path), f.contents))
    files.append(File(Path(relative_root, 'PKG-INFO'), f"Metadata-Version: {DEFAULT_METADATA_VERSION}\nName: {kwargs['project_name']}\nVersion: 0.0.1\nLicense-File: LICENSE.txt\nDescription-Content-Type: text/markdown\n\n# My.App\n\n[![PyPI - Version](https://img.shields.io/pypi/v/my-app.svg)](https://pypi.org/project/my-app)\n[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/my-app.svg)](https://pypi.org/project/my-app)\n\n-----\n\n**Table of Contents**\n\n- [Installation](#installation)\n- [License](#license)\n\n## Installation\n\n```console\npip install my-app\n```\n\n## License\n\n`my-app` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.\n"))
    return files