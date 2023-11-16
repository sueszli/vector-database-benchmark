"""Update the What's New doc (development version)

This collects the snippets from whatsnew/pr/, moves their content into
whatsnew/development.rst (chronologically ordered), and deletes the snippets.
"""
import io
import sys
from pathlib import Path
from subprocess import check_call, check_output
repo_root = Path(__file__).resolve().parent.parent
whatsnew_dir = repo_root / 'docs' / 'source' / 'whatsnew'
pr_dir = whatsnew_dir / 'pr'
target = whatsnew_dir / 'development.rst'
FEATURE_MARK = '.. DO NOT EDIT THIS LINE BEFORE RELEASE. FEATURE INSERTION POINT.'
INCOMPAT_MARK = '.. DO NOT EDIT THIS LINE BEFORE RELEASE. INCOMPAT INSERTION POINT.'
files = set(pr_dir.glob('*.rst'))
files.difference_update({pr_dir / f for f in {'incompat-switching-to-perl.rst', 'antigravity-feature.rst'}})
if not files:
    print("No automatic update available for what's new")
    sys.exit(0)

def getmtime(f):
    if False:
        return 10
    return check_output(['git', 'log', '-1', '--format="%ai"', '--', f])
files = sorted(files, key=getmtime)
(features, incompats) = ([], [])
for path in files:
    with io.open(path, encoding='utf-8') as f:
        try:
            content = f.read().rstrip()
        except Exception as e:
            raise Exception('Error reading "{}"'.format(f)) from e
    if path.name.startswith('incompat-'):
        incompats.append(content)
    else:
        features.append(content)
feature_block = '\n\n'.join(features + [FEATURE_MARK])
incompat_block = '\n\n'.join(incompats + [INCOMPAT_MARK])
with io.open(target, encoding='utf-8') as f:
    content = f.read()
assert content.count(FEATURE_MARK) == 1
assert content.count(INCOMPAT_MARK) == 1
content = content.replace(FEATURE_MARK, feature_block)
content = content.replace(INCOMPAT_MARK, incompat_block)
content = '\n'.join((l.rstrip() for l in content.splitlines()))
with io.open(target, 'w', encoding='utf-8') as f:
    f.write(content)
for file in files:
    check_call(['git', 'rm', file])
check_call(['git', 'add', target])
print("Merged what's new changes. Check the diff and commit the change.")