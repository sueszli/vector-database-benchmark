from hatch.template import File
from hatch.utils.fs import Path
from hatchling.__about__ import __version__
from hatchling.metadata.spec import DEFAULT_METADATA_VERSION
from ..new.feature_no_src_layout import get_files as get_template_files
from .utils import update_record_file_contents

def get_files(**kwargs):
    if False:
        i = 10
        return i + 15
    metadata_directory = kwargs.get('metadata_directory', '')
    package_root = kwargs.get('package_root', '')
    files = []
    for f in get_template_files(**kwargs):
        if str(f.path) == 'LICENSE.txt':
            files.append(File(Path(metadata_directory, 'licenses', f.path), f.contents))
        elif f.path.parts[-1] == '__about__.py':
            files.append(File(Path('zfoo.py'), f.contents))
    pth_file_name = f"_{kwargs['package_name']}.pth"
    loader_file_name = f"_editable_impl_{kwargs['package_name']}.py"
    files.extend((File(Path(pth_file_name), f"import _editable_impl_{kwargs['package_name']}"), File(Path(loader_file_name), f"from editables.redirector import RedirectingFinder as F\nF.install()\nF.map_module({kwargs['package_name']!r}, {package_root!r})"), File(Path(metadata_directory, 'WHEEL'), f'Wheel-Version: 1.0\nGenerator: hatchling {__version__}\nRoot-Is-Purelib: true\nTag: py2-none-any\nTag: py3-none-any\n'), File(Path(metadata_directory, 'METADATA'), f"Metadata-Version: {DEFAULT_METADATA_VERSION}\nName: {kwargs['project_name']}\nVersion: 0.0.1\nLicense-File: LICENSE.txt\nRequires-Dist: editables~=0.3\n")))
    record_file = File(Path(metadata_directory, 'RECORD'), '')
    update_record_file_contents(record_file, files, generated_files={pth_file_name, loader_file_name})
    files.append(record_file)
    return files