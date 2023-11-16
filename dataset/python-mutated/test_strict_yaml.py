import os
import pytest
from ruamel.yaml import __with_libyaml__ as ruamel_clib
from dvc.cli import main
DUPLICATE_KEYS = 'stages:\n  stage1:\n    cmd: python train.py\n    cmd: python train.py\n'
DUPLICATE_KEYS_OUTPUT = '\'./dvc.yaml\' is invalid.\n\nWhile constructing a mapping, in line 3, column 5\n  3 │   cmd: python train.py\n\nFound duplicate key "cmd" with value "python train.py" (original value: "python\ntrain.py"), in line 4, column 5\n  4 │   cmd: python train.py'
MAPPING_VALUES_NOT_ALLOWED = 'stages:\n  stage1\n    cmd: python script.py\n'
MAPPING_VALUES_NOT_ALLOWED_OUTPUT = "'./dvc.yaml' is invalid.\n\nMapping values are not allowed {}, in line 3, column 8\n  3 │   cmd: python script.py".format('in this context' if ruamel_clib else 'here')
NO_HYPHEN_INDICATOR_IN_BLOCK = 'stages:\n  stage1:\n    cmd: python script.py\n    outs:\n      - logs:\n          cache: false\n      metrics:\n'
NO_HYPHEN_INDICATOR_IN_BLOCK_OUTPUT = "'./dvc.yaml' is invalid.\n\nWhile parsing a block collection, in line 5, column 7\n  5 │     - logs:\n\n{}, in line 7, column 7\n  7 │     metrics:".format("Did not find expected '-' indicator" if ruamel_clib else "Expected <block end>, but found '?'")
UNCLOSED_SCALAR = 'stages:\n  stage1:\n    cmd: python script.py\n    desc: "this is my stage one\n'
UNCLOSED_SCALAR_OUTPUT = '\'./dvc.yaml\' is invalid.\n\nWhile scanning a quoted scalar, in line 4, column 11\n  4 │   desc: "this is my stage one\n\nFound unexpected end of stream, in line 5, column 1\n  5'
NOT_A_DICT = '3'
NOT_A_DICT_OUTPUT = "'./dvc.yaml' validation failed: expected a dictionary.\n"
EMPTY_STAGE = 'stages:\n  stage1:\n'
EMPTY_STAGE_OUTPUT = "'./dvc.yaml' validation failed.\n\nexpected a dictionary, in stages -> stage1, line 2, column 3\n  1 stages:\n  2   stage1:\n  3"
MISSING_CMD = 'stages:\n  stage1:\n    cmd: {}\n'
MISSING_CMD_OUTPUT = "'./dvc.yaml' validation failed.\n\nexpected str, in stages -> stage1 -> cmd, line 3, column 10\n  2   stage1:\n  3 │   cmd: {}"
DEPS_AS_DICT = 'stages:\n  stage1:\n    cmd: python script.py\n    deps:\n      - src:\n'
DEPS_AS_DICT_OUTPUT = "'./dvc.yaml' validation failed.\n\nexpected str, in stages -> stage1 -> deps -> 0, line 5, column 9\n  4 │   deps:\n  5 │     - src:\n"
OUTS_AS_STR = 'stages:\n  train:\n    cmd:\n      - python train.py\n    deps:\n      - config.cfg\n    outs:\n      models/'
OUTS_AS_STR_OUTPUT = "'./dvc.yaml' validation failed.\n\nexpected a list, in stages -> train -> outs, line 3, column 5\n  2   train:\n  3 │   cmd:\n  4 │     - python train.py"
NULL_VALUE_ON_OUTS = 'stages:\n  stage1:\n    cmd: python script.py\n    outs:\n    - logs:\n        cache: false\n        persist: true\n        remote:\n'
NULL_VALUE_ON_OUTS_OUTPUT = "'./dvc.yaml' validation failed.\n\nexpected str, in stages -> stage1 -> outs -> 0 -> logs -> remote, line 6, column\n9\n  5 │   - logs:\n  6 │   │   cache: false\n  7 │   │   persist: true"
ADDITIONAL_KEY_ON_OUTS = 'stages:\n  stage1:\n    cmd: python script.py\n    outs:\n    - logs:\n        cache: false\n        not_existing_key: false\n'
ADDITIONAL_KEY_ON_OUTS_OUTPUT = "'./dvc.yaml' validation failed.\n\nextra keys not allowed, in stages -> stage1 -> outs -> 0 -> logs ->\nnot_existing_key, line 6, column 9\n  5 │   - logs:\n  6 │   │   cache: false\n  7 │   │   not_existing_key: false"
FOREACH_SCALAR_VALUE = 'stages:\n  group:\n    foreach: 3\n    do:\n      cmd: python script${i}.py\n'
FOREACH_SCALAR_VALUE_OUTPUT = "'./dvc.yaml' validation failed.\n\nexpected dict, in stages -> group -> foreach, line 3, column 5\n  2   group:\n  3 │   foreach: 3\n  4 │   do:"
FOREACH_DO_NULL = 'stages:\n  stage1:\n    foreach: [1,2,3]\n    do:\n'
FOREACH_DO_NULL_OUTPUT = "'./dvc.yaml' validation failed.\n\nexpected a dictionary, in stages -> stage1 -> do, line 3, column 5\n  2   stage1:\n  3 │   foreach: [1,2,3]\n  4 │   do:"
FOREACH_DO_MISSING_CMD = 'stages:\n  stage1:\n    foreach: [1,2,3]\n    do:\n      outs:\n      - ${item}\n'
FOREACH_WITH_CMD_DO_MISSING = 'stages:\n  stage1:\n    foreach: [1,2,3]\n    cmd: python script${item}.py\n'
FOREACH_WITH_CMD_DO_MISSING_OUTPUT = "'./dvc.yaml' validation failed: 2 errors.\n\nextra keys not allowed, in stages -> stage1 -> cmd, line 3, column 5\n  2   stage1:\n  3 │   foreach: [1,2,3]\n  4 │   cmd: python script${item}.py\n\nrequired key not provided, in stages -> stage1 -> do, line 3, column 5\n  2   stage1:\n  3 │   foreach: [1,2,3]\n  4 │   cmd: python script${item}.py"
FOREACH_DO_MISSING_CMD_OUTPUT = "'./dvc.yaml' validation failed.\n\nrequired key not provided, in stages -> stage1 -> do -> cmd, line 5, column 7\n  4 │   do:\n  5 │     outs:\n  6 │     - ${item}"
MERGE_CONFLICTS = 'stages:\n  load_data:\n<<<<<<< HEAD\n    cmd: python src/load_data.py\n    deps:\n    - src/load_data.py\n=======\n    cmd: python load_data.py\n    deps:\n    - load_data.py\n>>>>>>> branch\n    outs:\n    - data\n'
MERGE_CONFLICTS_OUTPUT = "'./dvc.yaml' is invalid (possible merge conflicts).\n\nWhile scanning a simple key, in line 3, column 1\n  3 <<<<<<< HEAD\n\nCould not find expected ':', in line 4, column 8\n  4 │   cmd: python src/load_data.py"
examples = {'duplicate_keys': (DUPLICATE_KEYS, DUPLICATE_KEYS_OUTPUT), 'mapping_values_not_allowed': (MAPPING_VALUES_NOT_ALLOWED, MAPPING_VALUES_NOT_ALLOWED_OUTPUT), 'no_hyphen_block': (NO_HYPHEN_INDICATOR_IN_BLOCK, NO_HYPHEN_INDICATOR_IN_BLOCK_OUTPUT), 'unclosed_scalar': (UNCLOSED_SCALAR, UNCLOSED_SCALAR_OUTPUT), 'not_a_dict': (NOT_A_DICT, NOT_A_DICT_OUTPUT), 'empty_stage': (EMPTY_STAGE, EMPTY_STAGE_OUTPUT), 'missing_cmd': (MISSING_CMD, MISSING_CMD_OUTPUT), 'deps_as_dict': (DEPS_AS_DICT, DEPS_AS_DICT_OUTPUT), 'outs_as_str': (OUTS_AS_STR, OUTS_AS_STR_OUTPUT), 'null_value_on_outs': (NULL_VALUE_ON_OUTS, NULL_VALUE_ON_OUTS_OUTPUT), 'additional_key_on_outs': (ADDITIONAL_KEY_ON_OUTS, ADDITIONAL_KEY_ON_OUTS_OUTPUT), 'foreach_scalar': (FOREACH_SCALAR_VALUE, FOREACH_SCALAR_VALUE_OUTPUT), 'foreach_do_do_null': (FOREACH_DO_NULL, FOREACH_DO_NULL_OUTPUT), 'foreach_do_missing_cmd': (FOREACH_DO_MISSING_CMD, FOREACH_DO_MISSING_CMD_OUTPUT), 'foreach_unknown_cmd_missing_do': (FOREACH_WITH_CMD_DO_MISSING, FOREACH_WITH_CMD_DO_MISSING_OUTPUT), 'merge_conflicts': (MERGE_CONFLICTS, MERGE_CONFLICTS_OUTPUT)}

@pytest.fixture
def force_posixpath(mocker):
    if False:
        i = 10
        return i + 15
    mocker.patch('dvc.utils.strictyaml.make_relpath', return_value='./dvc.yaml')

@pytest.fixture
def fixed_width_term(mocker):
    if False:
        print('Hello World!')
    'Fixed width console.'
    from rich.console import Console
    mocker.patch.object(Console, 'width', new_callable=mocker.PropertyMock(return_value=80))

@pytest.mark.parametrize('text, expected', examples.values(), ids=examples.keys())
def test_exceptions(tmp_dir, dvc, capsys, force_posixpath, fixed_width_term, text, expected):
    if False:
        while True:
            i = 10
    tmp_dir.gen('dvc.yaml', text)
    capsys.readouterr()
    assert main(['stage', 'list']) != 0
    (out, err) = capsys.readouterr()
    assert not out
    for (expected_line, err_line) in zip(expected.splitlines(), err.splitlines()):
        assert expected_line == err_line.rstrip(' ')

@pytest.mark.parametrize('text, expected', [(DUPLICATE_KEYS, "'./dvc.yaml' is invalid in revision '{short_rev}'."), (MISSING_CMD, "'./dvc.yaml' validation failed in revision '{short_rev}'.")])
def test_on_revision(tmp_dir, scm, dvc, force_posixpath, fixed_width_term, capsys, text, expected):
    if False:
        while True:
            i = 10
    tmp_dir.scm_gen('dvc.yaml', text, commit='add dvc.yaml')
    capsys.readouterr()
    assert main(['ls', f'file://{tmp_dir.as_posix()}', '--rev', 'HEAD']) != 0
    (out, err) = capsys.readouterr()
    assert not out
    assert expected.format(short_rev=scm.get_rev()[:7]) in err

def test_make_relpath(tmp_dir, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    from dvc.utils.strictyaml import make_relpath
    path = tmp_dir / 'dvc.yaml'
    expected_path = './dvc.yaml' if os.name == 'posix' else '.\\dvc.yaml'
    assert make_relpath(path) == expected_path
    (tmp_dir / 'dir').mkdir(exist_ok=True)
    monkeypatch.chdir('dir')
    expected_path = '../dvc.yaml' if os.name == 'posix' else '..\\dvc.yaml'
    assert make_relpath(path) == expected_path

def test_fallback_exception_message(tmp_dir, dvc, mocker, caplog):
    if False:
        for i in range(10):
            print('nop')
    mocker.patch('dvc.utils.strictyaml.YAMLSyntaxError.__pretty_exc__', side_effect=ValueError)
    mocker.patch('dvc.utils.strictyaml.YAMLValidationError.__pretty_exc__', side_effect=ValueError)
    dvc_file = tmp_dir / 'dvc.yaml'
    dvc_file.write_text(MAPPING_VALUES_NOT_ALLOWED)
    assert main(['stage', 'list']) != 0
    assert "unable to read: 'dvc.yaml', YAML file structure is corrupted" in caplog.text
    caplog.clear()
    dvc_file.dump({'stages': {'stage1': None}})
    assert main(['stage', 'list']) != 0
    assert "dvc.yaml' validation failed" in caplog.text