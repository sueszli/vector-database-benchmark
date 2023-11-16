from hatch.template import File
from hatch.utils.fs import Path
from hatchling.metadata.spec import DEFAULT_METADATA_VERSION
from ..new.feature_no_src_layout import get_files as get_template_files

def get_files(**kwargs):
    if False:
        i = 10
        return i + 15
    relative_root = kwargs.get('relative_root', '')
    files = [File(Path(relative_root, f.path), f.contents) for f in get_template_files(**kwargs)]
    files.extend((File(Path(relative_root, kwargs['package_name'], 'lib.so'), ''), File(Path(relative_root, '.hgignore'), 'syntax: glob\n*.pyc\n\nsyntax: foo\nREADME.md\n\nsyntax: glob\n*.so\n*.h\n'), File(Path(relative_root, 'PKG-INFO'), f"Metadata-Version: {DEFAULT_METADATA_VERSION}\nName: {kwargs['project_name']}\nVersion: 0.0.1\nLicense-File: LICENSE.txt\n")))
    return files