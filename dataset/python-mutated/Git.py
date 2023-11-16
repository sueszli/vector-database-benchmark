""" Functions to handle git staged content.

Inspired from https://raw.githubusercontent.com/hallettj/git-format-staged/master/git-format-staged

Original author: Jesse Hallett <jesse@sitr.us>

"""
import os
import re
from nuitka.Tracing import my_print
from nuitka.utils.Execution import NuitkaCalledProcessError, check_call, check_output, executeProcess
from nuitka.utils.FileOperations import openTextFile

def _parseIndexDiffLine(line):
    if False:
        while True:
            i = 10
    pattern = re.compile('^:(\\d+) (\\d+) ([a-f0-9]+) ([a-f0-9]+) ([A-Z])(\\d+)?\\t([^\\t]+)(?:\\t([^\\t]+))?$')
    zeroed_pat = re.compile('^0+$')

    def unless_zeroed(s):
        if False:
            for i in range(10):
                print('nop')
        return s if not zeroed_pat.match(s) else None
    match = pattern.match(line)
    if not match:
        raise ValueError('Failed to parse diff-index line: ' + line)
    return {'src_mode': unless_zeroed(match.group(1)), 'dst_mode': unless_zeroed(match.group(2)), 'src_hash': unless_zeroed(match.group(3)), 'dst_hash': unless_zeroed(match.group(4)), 'status': match.group(5), 'score': int(match.group(6)) if match.group(6) else None, 'src_path': match.group(7), 'dst_path': match.group(8)}

def getStagedFileChangeDesc():
    if False:
        return 10
    output = check_output(['git', 'diff-index', '--cached', '--diff-filter=AM', '--no-renames', 'HEAD'])
    for line in output.splitlines():
        if str is not bytes:
            line = line.decode('utf8')
        yield _parseIndexDiffLine(line)

def getModifiedPaths():
    if False:
        for i in range(10):
            print('nop')
    result = set()
    output = check_output(['git', 'diff', '--name-only'])
    for line in output.splitlines():
        if str is not bytes:
            line = line.decode('utf8')
        result.add(line)
    output = check_output(['git', 'diff', '--cached', '--name-only'])
    for line in output.splitlines():
        if str is not bytes:
            line = line.decode('utf8')
        result.add(line)
    return tuple(sorted(result))

def getUnpushedPaths():
    if False:
        while True:
            i = 10
    result = set()
    try:
        output = check_output(['git', 'diff', '--stat', '--name-only', '@{upstream}'])
    except NuitkaCalledProcessError:
        return result
    for line in output.splitlines():
        if str is not bytes:
            line = line.decode('utf8')
        if not os.path.exists(line):
            continue
        result.add(line)
    return tuple(sorted(result))

def getFileHashContent(object_hash):
    if False:
        i = 10
        return i + 15
    return check_output(['git', 'cat-file', '-p', object_hash])

def putFileHashContent(filename):
    if False:
        while True:
            i = 10
    with openTextFile(filename, 'r') as input_file:
        new_hash = check_output(['git', 'hash-object', '-w', '--stdin'], stdin=input_file)
    if str is not bytes:
        new_hash = new_hash.decode('utf8')
    assert new_hash
    return new_hash.rstrip()

def updateFileIndex(diff_entry, new_object_hash):
    if False:
        while True:
            i = 10
    check_call(['git', 'update-index', '--cacheinfo', '%s,%s,%s' % (diff_entry['dst_mode'], new_object_hash, diff_entry['src_path'])])

def updateWorkingFile(path, orig_object_hash, new_object_hash):
    if False:
        return 10
    patch = check_output(['git', 'diff', '--no-color', orig_object_hash, new_object_hash])
    git_path = path.replace(os.path.sep, '/').encode('utf8')

    def updateLine(line):
        if False:
            print('Hello World!')
        if line.startswith(b'diff --git'):
            line = b'diff --git a/%s b/%s' % (git_path, git_path)
        elif line.startswith(b'--- a/'):
            line = b'--- a/' + git_path
        elif line.startswith(b'+++ b/'):
            line = b'+++ b/' + git_path
        return line
    patch = b'\n'.join((updateLine(line) for line in patch.splitlines())) + b'\n'
    (output, err, exit_code) = executeProcess(['git', 'apply', '-'], stdin=patch)
    if exit_code != 0 and os.name == 'nt':
        from .auto_format.AutoFormat import cleanupWindowsNewlines
        cleanupWindowsNewlines(path, path)
        (output, err, exit_code) = executeProcess(['git', 'apply', '-'], stdin=patch)
    success = exit_code == 0
    if not success:
        if output:
            my_print(output, style='yellow')
        if err:
            my_print(err, style='yellow')
    return success