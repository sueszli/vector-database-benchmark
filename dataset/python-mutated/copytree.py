import fnmatch
import os
from functools import partial
from pathlib import Path
from shutil import Error, copy2, copystat
from typing import Callable, List, Optional, Set, Tuple, Union
from lightning.app.core.constants import DOT_IGNORE_FILENAME
from lightning.app.utilities.app_helpers import Logger
logger = Logger(__name__)
_IGNORE_FUNCTION = Callable[[Path, List[Path]], List[Path]]

def _copytree(src: Union[Path, str], dst: Union[Path, str], ignore_functions: Optional[List[_IGNORE_FUNCTION]]=None, dirs_exist_ok=False, dry_run=False) -> List[str]:
    if False:
        print('Hello World!')
    "Vendor in from `shutil.copytree` to support ignoring files recursively based on `.lightningignore`, like `git`\n    does with `.gitignore`. Also removed a few checks from the original copytree related to symlink checks. Differences\n    between original and this function are.\n\n    1. It supports a list of ignore function instead of a single one in the\n        original. We can use this for filtering out files based on nested\n        .lightningignore files\n    2. It supports a dry run. When enabled, this function will not copy anything but just recursively\n        find the source files which are not-ignored and return them. It is useful while calculating\n        the hash or checking the size of files\n    3. This function returns a list of copied files unlike the original which was returning the\n        destination directory\n\n    Recursively copy a directory tree and return the destination directory.\n\n    Parameters\n    ----------\n    src:\n        Source directory path to copy from\n    dst:\n        Destination directory path to copy to\n    ignore_functions:\n        List of functions that will be used to filter out files\n        and directories. This isn't required to be passed when calling from outside but will be\n        autopopulated by the recursive calls in this function itself (Original copytree doesn't have this argument)\n    dirs_exist_ok:\n        If true, the destination directory will be created if it doesn't exist.\n    dry_run:\n        If true, this function will not copy anything (this is not present in the original copytree)\n\n\n    If exception(s) occur, an Error is raised with a list of reasons.\n\n    "
    files_copied = []
    if ignore_functions is None:
        ignore_functions = []
    _ignore_filename_spell_check(src)
    src = Path(src)
    dst = Path(dst)
    ignore_filepath = src / DOT_IGNORE_FILENAME
    if ignore_filepath.is_file():
        patterns = _read_lightningignore(ignore_filepath)
        ignore_fn = partial(_filter_ignored, src, patterns)
        ignore_functions = [*ignore_functions, ignore_fn]
    if not dry_run:
        os.makedirs(dst, exist_ok=dirs_exist_ok)
    errors = []
    entries = list(src.iterdir())
    for fn in ignore_functions:
        entries = fn(src, entries)
    for srcentry in entries:
        dstpath = dst / srcentry.name
        try:
            if srcentry.is_dir():
                _files = _copytree(src=srcentry, dst=dstpath, ignore_functions=ignore_functions, dirs_exist_ok=dirs_exist_ok, dry_run=dry_run)
                files_copied.extend(_files)
            else:
                files_copied.append(str(srcentry))
                if not dry_run:
                    copy2(srcentry, dstpath)
        except Error as err:
            errors.extend(err.args[0])
        except OSError as why:
            errors.append((srcentry, dstpath, str(why)))
    try:
        if not dry_run:
            copystat(src, dst)
    except OSError as why:
        if getattr(why, 'winerror', None) is None:
            errors.append((src, dst, str(why)))
    if errors:
        raise Error(errors)
    return files_copied

def _filter_ignored(src: Path, patterns: Set[str], current_dir: Path, entries: List[Path]) -> List[Path]:
    if False:
        i = 10
        return i + 15
    relative_dir = current_dir.relative_to(src)
    names = [str(relative_dir / entry.name) for entry in entries]
    ignored_names = set()
    for pattern in patterns:
        ignored_names.update(fnmatch.filter(names, pattern))
    return [entry for entry in entries if str(relative_dir / entry.name) not in ignored_names]

def _parse_lightningignore(lines: Tuple[str]) -> Set[str]:
    if False:
        return 10
    'Creates a set that removes empty lines and comments.'
    lines = [ln.strip() for ln in lines]
    lines = [ln.lstrip('/').lstrip('\\') for ln in lines if ln != '' and (not ln.startswith('#'))]
    return {str(Path(ln)) for ln in lines}

def _read_lightningignore(path: Path) -> Set[str]:
    if False:
        i = 10
        return i + 15
    "Reads ignore file and filter and empty lines. This will also remove patterns that start with a `/`. That's done\n    to allow `glob` to simulate the behavior done by `git` where it interprets that as a root path.\n\n    Parameters\n    ----------\n    path: Path\n        Path to .lightningignore file or equivalent.\n\n    Returns\n    -------\n    Set[str]\n        Set of unique lines.\n\n    "
    raw_lines = path.open().readlines()
    return _parse_lightningignore(raw_lines)

def _ignore_filename_spell_check(src: Path):
    if False:
        return 10
    possible_spelling_mistakes = ['.gridignore', '.lightingignore', '.lightinginore', '.lightninginore', '.lightninignore', '.lightinignore']
    possible_spelling_mistakes.extend([p.lstrip('.') for p in possible_spelling_mistakes])
    for path in src.iterdir():
        if path.is_file() and path.name in possible_spelling_mistakes:
            logger.warn(f'Lightning uses `{DOT_IGNORE_FILENAME}` as the ignore file but found {path.name} at {path.parent} instead. If this was a mistake, please rename the file.')