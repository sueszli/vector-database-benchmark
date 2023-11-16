import os

def find_files(substring, path, recursive=False, check_ext=None, ignore_invisible=True, ignore_substring=None):
    if False:
        print('Hello World!')
    "Find files in a directory based on substring matching.\n\n    Parameters\n    ----------\n    substring : `str`\n        Substring of the file to be matched.\n    path : `str`\n        Path where to look.\n    recursive : `bool`\n        If true, searches subdirectories recursively.\n    check_ext : `str`\n        If string (e.g., '.txt'), only returns files that\n        match the specified file extension.\n    ignore_invisible : `bool`\n        If `True`, ignores invisible files\n        (i.e., files starting with a period).\n    ignore_substring : `str`\n        Ignores files that contain the specified substring.\n\n    Returns\n    ----------\n    results : `list`\n        List of the matched files.\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/file_io/find_files/\n\n    "

    def check_file(f, path):
        if False:
            return 10
        if not (ignore_substring and ignore_substring in f):
            if substring in f:
                compl_path = os.path.join(path, f)
                if os.path.isfile(compl_path):
                    return compl_path
        return False
    results = []
    if recursive:
        for (par, nxt, fnames) in os.walk(path):
            for f in fnames:
                fn = check_file(f, par)
                if fn:
                    results.append(fn)
    else:
        for f in os.listdir(path):
            if ignore_invisible and f.startswith('.'):
                continue
            fn = check_file(f, path)
            if fn:
                results.append(fn)
    if check_ext:
        results = [r for r in results if os.path.splitext(r)[-1] == check_ext]
    return results