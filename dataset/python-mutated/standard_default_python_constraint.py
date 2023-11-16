from hatch.template import File
from hatch.utils.fs import Path
from hatchling.__about__ import __version__
from hatchling.metadata.spec import DEFAULT_METADATA_VERSION
from ..new.feature_no_src_layout import get_files as get_template_files
from .utils import update_record_file_contents

def get_files(**kwargs):
    if False:
        return 10
    metadata_directory = kwargs.get('metadata_directory', '')
    files = []
    for f in get_template_files(**kwargs):
        if str(f.path) == 'LICENSE.txt':
            files.append(File(Path(metadata_directory, 'licenses', f.path), f.contents))
        if f.path.parts[0] != kwargs['package_name']:
            continue
        files.append(f)
    files.extend((File(Path(metadata_directory, 'WHEEL'), f'Wheel-Version: 1.0\nGenerator: hatchling {__version__}\nRoot-Is-Purelib: true\nTag: py3-none-any\n'), File(Path(metadata_directory, 'METADATA'), f"Metadata-Version: {DEFAULT_METADATA_VERSION}\nName: {kwargs['project_name']}\nVersion: 0.0.1\nLicense-File: LICENSE.txt\nRequires-Python: >3\n")))
    record_file = File(Path(metadata_directory, 'RECORD'), '')
    update_record_file_contents(record_file, files)
    files.append(record_file)
    return files