"""Tests for Tutorial module."""
import unittest
import doctest
import os
import sys
import warnings
from Bio import BiopythonExperimentalWarning, MissingExternalDependencyError
import requires_internet
try:
    requires_internet.check()
    online = True
except MissingExternalDependencyError:
    online = False
if '--offline' in sys.argv:
    online = False
warnings.simplefilter('ignore', BiopythonExperimentalWarning)
original_path = os.path.abspath('.')
if os.path.basename(sys.argv[0]) == 'test_Tutorial.py':
    tutorial_base = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '../Doc/'))
    tutorial = os.path.join(tutorial_base, 'Tutorial.tex')
else:
    tutorial_base = os.path.abspath('../Doc/')
    tutorial = os.path.join(tutorial_base, 'Tutorial.tex')
if not os.path.isfile(tutorial):
    from Bio import MissingExternalDependencyError
    raise MissingExternalDependencyError('Could not find ../Doc/Tutorial.tex file')
files = [tutorial]
for latex in os.listdir(os.path.join(tutorial_base, 'Tutorial/')):
    if latex.startswith('chapter_') and latex.endswith('.tex'):
        files.append(os.path.join(tutorial_base, 'Tutorial', latex))

def _extract(handle):
    if False:
        i = 10
        return i + 15
    line = handle.readline()
    if line != '\\begin{minted}{pycon}\n':
        if not (line.startswith('\\begin{minted}[') and line.endswith(']{pycon}\n')):
            raise ValueError("Any '%doctest' or '%cont-doctest' line should be followed by '\\begin{minted}{pycon}' or '\\begin{minted}[options]{pycon}'")
    lines = []
    while True:
        line = handle.readline()
        if not line:
            if lines:
                print(''.join(lines[:30]))
                raise ValueError("Didn't find end of test starting: %r", lines[0])
            else:
                raise ValueError("Didn't find end of test!")
        elif line.startswith('\\end{minted}'):
            break
        else:
            lines.append(line)
    return lines

def extract_doctests(latex_filename):
    if False:
        for i in range(10):
            print('nop')
    'Scan LaTeX file and pull out marked doctests as strings.\n\n    This is a generator, yielding one tuple per doctest.\n    '
    base_name = os.path.splitext(os.path.basename(latex_filename))[0]
    deps = ''
    folder = ''
    with open(latex_filename) as handle:
        line_number = 0
        lines = []
        name = None
        while True:
            line = handle.readline()
            line_number += 1
            if not line:
                break
            elif line.startswith('%cont-doctest'):
                x = _extract(handle)
                lines.extend(x)
                line_number += len(x) + 2
            elif line.startswith('%doctest'):
                if lines:
                    if not lines[0].startswith('>>> '):
                        raise ValueError(f"Should start '>>> ' not {lines[0]!r}")
                    yield (name, ''.join(lines), folder, deps)
                    lines = []
                deps = [x.strip() for x in line.split()[1:]]
                if deps:
                    folder = deps[0]
                    deps = deps[1:]
                else:
                    folder = ''
                name = 'test_%s_line_%05i' % (base_name, line_number)
                x = _extract(handle)
                lines.extend(x)
                line_number += len(x) + 2
    if lines:
        if not lines[0].startswith('>>> '):
            raise ValueError(f"Should start '>>> ' not {lines[0]!r}")
        yield (name, ''.join(lines), folder, deps)

class TutorialDocTestHolder:
    """Python doctests extracted from the Biopython Tutorial."""

def check_deps(dependencies):
    if False:
        while True:
            i = 10
    "Check 'lib:XXX' and 'internet' dependencies are met."
    missing = []
    for dep in dependencies:
        if dep == 'internet':
            if not online:
                missing.append('internet')
        else:
            assert dep.startswith('lib:'), dep
            lib = dep[4:]
            try:
                tmp = __import__(lib)
                del tmp
            except ImportError:
                missing.append(lib)
    return missing
missing_deps = set()
for latex in files:
    for (name, example, folder, deps) in extract_doctests(latex):
        missing = check_deps(deps)
        if missing:
            missing_deps.update(missing)
            continue

        def funct(n, d, f):
            if False:
                return 10
            global tutorial_base
            method = lambda x: None
            if f:
                p = os.path.join(tutorial_base, f)
                method.__doc__ = f'{n}\n\n>>> import os\n>>> os.chdir({p!r})\n{d}\n'
            else:
                method.__doc__ = f'{n}\n\n{d}\n'
            method._folder = f
            return method
        setattr(TutorialDocTestHolder, f"doctest_{name.replace(' ', '_')}", funct(name, example, folder))
        del funct

class TutorialTestCase(unittest.TestCase):
    """Python doctests extracted from the Biopython Tutorial."""

    def test_doctests(self):
        if False:
            for i in range(10):
                print('nop')
        'Run tutorial doctests.'
        runner = doctest.DocTestRunner()
        failures = []
        for test in doctest.DocTestFinder().find(TutorialDocTestHolder):
            (failed, success) = runner.run(test)
            if failed:
                name = test.name
                assert name.startswith('TutorialDocTestHolder.doctest_')
                failures.append(name[30:])
        if failures:
            raise ValueError('%i Tutorial doctests failed: %s' % (len(failures), ', '.join(failures)))

    def tearDown(self):
        if False:
            return 10
        global original_path
        os.chdir(original_path)
        delete_phylo_tutorial = ['examples/tree1.nwk', 'examples/other_trees.xml']
        for file in delete_phylo_tutorial:
            if os.path.exists(os.path.join(tutorial_base, file)):
                os.remove(os.path.join(tutorial_base, file))
        tutorial_cluster_base = os.path.abspath('../Tests/')
        delete_cluster_tutorial = ['Cluster/cyano_result.atr', 'Cluster/cyano_result.cdt', 'Cluster/cyano_result.gtr', 'Cluster/cyano_result_K_A2.kag', 'Cluster/cyano_result_K_G5.kgg', 'Cluster/cyano_result_K_G5_A2.cdt']
        for file in delete_cluster_tutorial:
            if os.path.exists(os.path.join(tutorial_cluster_base, file)):
                os.remove(os.path.join(tutorial_cluster_base, file))
if __name__ == '__main__':
    if missing_deps:
        print('Skipping tests needing the following:')
        for dep in sorted(missing_deps):
            print(f' - {dep}')
    print('Running Tutorial doctests...')
    tests = doctest.testmod()
    if tests.failed:
        raise RuntimeError('%i/%i tests failed' % tests)
    print('Tests done')