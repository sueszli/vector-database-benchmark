from subprocess import check_output
import sys
import os.path

def main(tarname, gitroot):
    if False:
        for i in range(10):
            print('nop')
    'Run this as ./compare_tar_against_git.py TARFILE GITROOT\n\n    Args\n    ====\n\n    TARFILE: Path to the built sdist (sympy-xx.tar.gz)\n    GITROOT: Path ro root of git (dir containing .git)\n    '
    compare_tar_against_git(tarname, gitroot)
git_whitelist = {'.gitattributes', '.gitignore', '.mailmap', '.github/PULL_REQUEST_TEMPLATE.md', '.github/workflows/runtests.yml', '.github/workflows/ci-sage.yml', '.github/workflows/comment-on-pr.yml', '.github/workflows/release.yml', '.github/workflows/docs-preview.yml', '.github/workflows/checkconflict.yml', '.ci/durations.json', '.ci/generate_durations_log.sh', '.ci/parse_durations_log.py', '.ci/blacklisted.json', '.ci/README.rst', '.circleci/config.yml', '.github/FUNDING.yml', '.editorconfig', '.coveragerc', '.flake8', 'CODEOWNERS', 'asv.conf.actions.json', 'codecov.yml', 'requirements-dev.txt', 'MANIFEST.in', 'banner.svg', 'CODE_OF_CONDUCT.md', 'CONTRIBUTING.md', 'CITATION.cff', 'bin/adapt_paths.py', 'bin/ask_update.py', 'bin/authors_update.py', 'bin/build_doc.sh', 'bin/coverage_doctest.py', 'bin/coverage_report.py', 'bin/deploy_doc.sh', 'bin/diagnose_imports', 'bin/doctest', 'bin/generate_module_list.py', 'bin/generate_test_list.py', 'bin/get_sympy.py', 'bin/mailmap_update.py', 'bin/py.bench', 'bin/strip_whitespace', 'bin/sympy_time.py', 'bin/sympy_time_cache.py', 'bin/test', 'bin/test_external_imports.py', 'bin/test_executable.py', 'bin/test_import', 'bin/test_import.py', 'bin/test_isolated', 'bin/test_py2_import.py', 'bin/test_setup.py', 'bin/test_submodule_imports.py', 'bin/test_optional_dependencies.py', 'bin/test_sphinx.sh', 'bin/mailmap_check.py', 'bin/test_symengine.py', 'bin/test_tensorflow.py', 'bin/test_pyodide.mjs', 'examples/advanced/identitysearch_example.ipynb', 'examples/beginner/plot_advanced.ipynb', 'examples/beginner/plot_colors.ipynb', 'examples/beginner/plot_discont.ipynb', 'examples/beginner/plot_gallery.ipynb', 'examples/beginner/plot_intro.ipynb', 'examples/intermediate/limit_examples_advanced.ipynb', 'examples/intermediate/schwarzschild.ipynb', 'examples/notebooks/density.ipynb', 'examples/notebooks/fidelity.ipynb', 'examples/notebooks/fresnel_integrals.ipynb', 'examples/notebooks/qubits.ipynb', 'examples/notebooks/sho1d_example.ipynb', 'examples/notebooks/spin.ipynb', 'examples/notebooks/trace.ipynb', 'examples/notebooks/Bezout_Dixon_resultant.ipynb', 'examples/notebooks/IntegrationOverPolytopes.ipynb', 'examples/notebooks/Macaulay_resultant.ipynb', 'examples/notebooks/Sylvester_resultant.ipynb', 'examples/notebooks/README.txt', 'release/.gitignore', 'release/README.md', 'release/compare_tar_against_git.py', 'release/update_docs.py', 'release/build_docs.py', 'release/github_release.py', 'release/helpers.py', 'release/releasecheck.py', 'release/sha256.py', 'release/authors.py', 'release/ci_release_script.sh', 'conftest.py'}
tarball_whitelist = {'PKG-INFO', 'setup.cfg', 'sympy.egg-info/PKG-INFO', 'sympy.egg-info/SOURCES.txt', 'sympy.egg-info/dependency_links.txt', 'sympy.egg-info/requires.txt', 'sympy.egg-info/top_level.txt', 'sympy.egg-info/not-zip-safe', 'sympy.egg-info/entry_points.txt', 'doc/commit_hash.txt'}

def blue(text):
    if False:
        i = 10
        return i + 15
    return '\x1b[34m%s\x1b[0m' % text

def red(text):
    if False:
        for i in range(10):
            print('nop')
    return '\x1b[31m%s\x1b[0m' % text

def run(*cmdline, cwd=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Run command in subprocess and get lines of output\n    '
    return check_output(cmdline, encoding='utf-8', cwd=cwd).splitlines()

def full_path_split(path):
    if False:
        return 10
    '\n    Function to do a full split on a path.\n    '
    (rest, tail) = os.path.split(path)
    if not rest or rest == os.path.sep:
        return (tail,)
    return full_path_split(rest) + (tail,)

def compare_tar_against_git(tarname, gitroot):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compare the contents of the tarball against git ls-files\n\n    See the bottom of the file for the whitelists.\n    '
    git_lsfiles = {i.strip() for i in run('git', 'ls-files', cwd=gitroot)}
    tar_output_orig = set(run('tar', 'tf', tarname))
    tar_output = set()
    for file in tar_output_orig:
        split_path = full_path_split(file)
        if split_path[-1]:
            tar_output.add(os.path.join(*split_path[1:]))
    fail = False
    print()
    print(blue('Files in the tarball from git that should not be there:'))
    print()
    for line in sorted(tar_output.intersection(git_whitelist)):
        fail = True
        print(line)
    print()
    print(blue('Files in git but not in the tarball:'))
    print()
    for line in sorted(git_lsfiles - tar_output - git_whitelist):
        fail = True
        print(line)
    print()
    print(blue('Files in the tarball but not in git:'))
    print()
    for line in sorted(tar_output - git_lsfiles - tarball_whitelist):
        fail = True
        print(line)
    print()
    if fail:
        sys.exit(red('Non-whitelisted files found or not found in the tarball'))
if __name__ == '__main__':
    main(*sys.argv[1:])