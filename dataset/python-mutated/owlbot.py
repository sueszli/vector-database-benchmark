from logging import Logger
from pathlib import Path
import re
import subprocess
import synthtool as s
import synthtool.gcp as gcp
from synthtool.log import logger
logger: Logger = logger
_EXCLUDED_DIRS = ['^\\.']

def walk_through_owlbot_dirs(dir: Path, search_for_changed_files: bool) -> list[str]:
    if False:
        while True:
            i = 10
    '\n    Walks through all sample directories\n    Returns:\n    A list of directories\n    '
    owlbot_dirs: list[str] = []
    packages_to_exclude = _EXCLUDED_DIRS
    if search_for_changed_files:
        try:
            output = subprocess.run(['git', 'fetch', 'origin', 'main:main', '--deepen=200'], check=False)
            output.check_returncode()
        except subprocess.CalledProcessError as error:
            if error.returncode == 128:
                logger.info(f'Error: ${error.output}; skipping fetching main')
            else:
                raise error
    for path_object in dir.glob('**/requirements.txt'):
        object_dir = str(Path(path_object).parents[0])
        if path_object.is_file() and object_dir != str(dir) and (not re.search('(?:% s)' % '|'.join(packages_to_exclude), str(Path(path_object)))):
            if search_for_changed_files:
                if subprocess.run(['git', 'diff', '--quiet', 'main...', object_dir], check=False).returncode == 1:
                    owlbot_dirs.append(object_dir)
            else:
                owlbot_dirs.append(object_dir)
    for path_object in dir.glob('owl-bot-staging/*'):
        owlbot_dirs.append(f'{Path(path_object).parents[1]}/packages/{Path(path_object).name}')
    return owlbot_dirs
templated_files = gcp.CommonTemplates().py_library()
s.move(templated_files / 'noxfile.py')
dirs: list[str] = walk_through_owlbot_dirs(Path.cwd(), search_for_changed_files=True)
if dirs:
    lint_paths = ', '.join((f'"{d}"' for d in dirs))
    s.replace('noxfile.py', 'LINT_PATHS = \\["docs", "google", "tests", "noxfile.py", "setup.py"\\]', f'LINT_PATHS = [{lint_paths}]')
    s.replace('noxfile.py', 'BLACK_VERSION = "black==22.3.0"\\nISORT_VERSION = "isort==5.10.1', 'BLACK_VERSION = "black[jupyter]==23.3.0"\\nISORT_VERSION = "isort==5.11.0')
    s.shell.run(['nox', '-s', 'blacken'], hide_output=False)