import os
import re
from . import find_files

def find_filegroups(paths, substring='', extensions=None, validity_check=True, ignore_invisible=True, rstrip='', ignore_substring=None):
    if False:
        while True:
            i = 10
    'Find and collect files from different directories in a python dictionary.\n\n    Parameters\n    ----------\n    paths : `list`\n        Paths of the directories to be searched. Dictionary keys are build from\n        the first directory.\n    substring : `str` (default: \'\')\n        Substring that all files have to contain to be considered.\n    extensions : `list` (default: None)\n        `None` or `list` of allowed file extensions for each path.\n        If provided, the number of extensions must match the number of `paths`.\n    validity_check : `bool` (default: None)\n        If `True`, checks if all dictionary values\n        have the same number of file paths. Prints\n        a warning and returns an empty dictionary if the validity check failed.\n    ignore_invisible : `bool` (default: True)\n        If `True`, ignores invisible files\n        (i.e., files starting with a period).\n    rstrip : `str` (default: \'\')\n        If provided, strips characters from right side of the file\n        base names after splitting the extension.\n        Useful to trim different filenames to a common stem.\n        E.g,. "abc_d.txt" and "abc_d_.csv" would share\n        the stem "abc_d" if rstrip is set to "_".\n    ignore_substring : `str` (default: None)\n        Ignores files that contain the specified substring.\n\n    Returns\n    ----------\n    groups : `dict`\n        Dictionary of files paths. Keys are the file names\n        found in the first directory listed\n        in `paths` (without file extension).\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/file_io/find_filegroups/\n\n    '
    n = len(paths)
    assert len(paths) >= 2
    if extensions:
        assert len(extensions) == n
    else:
        extensions = ['' for i in range(n)]
    base = find_files(path=paths[0], substring=substring, check_ext=extensions[0], ignore_invisible=ignore_invisible, ignore_substring=ignore_substring)
    rest = [find_files(path=paths[i], substring=substring, check_ext=extensions[i], ignore_invisible=ignore_invisible, ignore_substring=ignore_substring) for i in range(1, n)]
    groups = {}
    for f in base:
        basename = os.path.splitext(os.path.basename(f))[0]
        basename = re.sub('\\%s$' % rstrip, '', basename)
        groups[basename] = [f]
    for (idx, r) in enumerate(rest):
        for f in r:
            (basename, ext) = os.path.splitext(os.path.basename(f))
            basename = re.sub('\\%s$' % rstrip, '', basename)
            try:
                if extensions[idx + 1] == '' or ext == extensions[idx + 1]:
                    groups[basename].append(f)
            except KeyError:
                pass
    if validity_check:
        lens = [len(groups[k]) for k in groups.keys()]
        if len(set(lens)) > 1:
            raise ValueError('Warning, some keys have more/less values than others. Set validity_check=False to ignore this warning.')
    return groups