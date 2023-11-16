from __future__ import annotations
import os.path
import shutil
import pytest
from pre_commit import envcontext
from pre_commit.languages import r
from pre_commit.prefix import Prefix
from pre_commit.store import _make_local_repo
from pre_commit.util import win_exe
from testing.language_helpers import run_language

def test_r_parsing_file_no_opts_no_args(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    cmd = r._cmd_from_hook(Prefix(str(tmp_path)), 'Rscript some-script.R', (), is_local=False)
    assert cmd == ('Rscript', '--no-save', '--no-restore', '--no-site-file', '--no-environ', str(tmp_path.joinpath('some-script.R')))

def test_r_parsing_file_opts_no_args():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError) as excinfo:
        r._entry_validate(['Rscript', '--no-init', '/path/to/file'])
    (msg,) = excinfo.value.args
    assert msg == 'The only valid syntax is `Rscript -e {expr}`or `Rscript path/to/hook/script`'

def test_r_parsing_file_no_opts_args(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    cmd = r._cmd_from_hook(Prefix(str(tmp_path)), 'Rscript some-script.R', ('--no-cache',), is_local=False)
    assert cmd == ('Rscript', '--no-save', '--no-restore', '--no-site-file', '--no-environ', str(tmp_path.joinpath('some-script.R')), '--no-cache')

def test_r_parsing_expr_no_opts_no_args1(tmp_path):
    if False:
        i = 10
        return i + 15
    cmd = r._cmd_from_hook(Prefix(str(tmp_path)), "Rscript -e '1+1'", (), is_local=False)
    assert cmd == ('Rscript', '--no-save', '--no-restore', '--no-site-file', '--no-environ', '-e', '1+1')

def test_r_parsing_local_hook_path_is_not_expanded(tmp_path):
    if False:
        while True:
            i = 10
    cmd = r._cmd_from_hook(Prefix(str(tmp_path)), 'Rscript path/to/thing.R', (), is_local=True)
    assert cmd == ('Rscript', '--no-save', '--no-restore', '--no-site-file', '--no-environ', 'path/to/thing.R')

def test_r_parsing_expr_no_opts_no_args2():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError) as excinfo:
        r._entry_validate(['Rscript', '-e', '1+1', '-e', 'letters'])
    (msg,) = excinfo.value.args
    assert msg == 'You can supply at most one expression.'

def test_r_parsing_expr_opts_no_args2():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError) as excinfo:
        r._entry_validate(['Rscript', '--vanilla', '-e', '1+1', '-e', 'letters'])
    (msg,) = excinfo.value.args
    assert msg == 'The only valid syntax is `Rscript -e {expr}`or `Rscript path/to/hook/script`'

def test_r_parsing_expr_args_in_entry2():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError) as excinfo:
        r._entry_validate(['Rscript', '-e', 'expr1', '--another-arg'])
    (msg,) = excinfo.value.args
    assert msg == 'You can supply at most one expression.'

def test_r_parsing_expr_non_Rscirpt():
    if False:
        return 10
    with pytest.raises(ValueError) as excinfo:
        r._entry_validate(['AnotherScript', '-e', '{{}}'])
    (msg,) = excinfo.value.args
    assert msg == 'entry must start with `Rscript`.'

def test_rscript_exec_relative_to_r_home():
    if False:
        i = 10
        return i + 15
    expected = os.path.join('r_home_dir', 'bin', win_exe('Rscript'))
    with envcontext.envcontext((('R_HOME', 'r_home_dir'),)):
        assert r._rscript_exec() == expected

def test_path_rscript_exec_no_r_home_set():
    if False:
        return 10
    with envcontext.envcontext((('R_HOME', envcontext.UNSET),)):
        assert r._rscript_exec() == 'Rscript'

def test_r_hook(tmp_path):
    if False:
        print('Hello World!')
    renv_lock = '{\n  "R": {\n    "Version": "4.0.3",\n    "Repositories": [\n      {\n        "Name": "CRAN",\n        "URL": "https://cloud.r-project.org"\n      }\n    ]\n  },\n  "Packages": {\n    "renv": {\n      "Package": "renv",\n      "Version": "0.12.5",\n      "Source": "Repository",\n      "Repository": "CRAN",\n      "Hash": "5c0cdb37f063c58cdab3c7e9fbb8bd2c"\n    },\n    "rprojroot": {\n      "Package": "rprojroot",\n      "Version": "1.0",\n      "Source": "Repository",\n      "Repository": "CRAN",\n      "Hash": "86704667fe0860e4fec35afdfec137f3"\n    }\n  }\n}\n'
    description = 'Package: gli.clu\nTitle: What the Package Does (One Line, Title Case)\nType: Package\nVersion: 0.0.0.9000\nAuthors@R:\n    person(given = "First",\n           family = "Last",\n           role = c("aut", "cre"),\n           email = "first.last@example.com",\n           comment = c(ORCID = "YOUR-ORCID-ID"))\nDescription: What the package does (one paragraph).\nLicense: `use_mit_license()`, `use_gpl3_license()` or friends to\n    pick a license\nEncoding: UTF-8\nLazyData: true\nRoxygen: list(markdown = TRUE)\nRoxygenNote: 7.1.1\nImports:\n    rprojroot\n'
    hello_world_r = 'stopifnot(\n    packageVersion(\'rprojroot\') == \'1.0\',\n    packageVersion(\'gli.clu\') == \'0.0.0.9000\'\n)\ncat("Hello, World, from R!\n")\n'
    tmp_path.joinpath('renv.lock').write_text(renv_lock)
    tmp_path.joinpath('DESCRIPTION').write_text(description)
    tmp_path.joinpath('hello-world.R').write_text(hello_world_r)
    renv_dir = tmp_path.joinpath('renv')
    renv_dir.mkdir()
    shutil.copy(os.path.join(os.path.dirname(__file__), '../../pre_commit/resources/empty_template_activate.R'), renv_dir.joinpath('activate.R'))
    expected = (0, b'Hello, World, from R!\n')
    assert run_language(tmp_path, r, 'Rscript hello-world.R') == expected

def test_r_inline(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    _make_local_repo(str(tmp_path))
    cmd = 'Rscript -e \'\n    stopifnot(packageVersion("rprojroot") == "1.0")\n    cat(commandArgs(trailingOnly = TRUE), "from R!\n", sep=", ")\n\'\n'
    ret = run_language(tmp_path, r, cmd, deps=('rprojroot@1.0',), args=('hi', 'hello'))
    assert ret == (0, b'hi, hello, from R!\n')