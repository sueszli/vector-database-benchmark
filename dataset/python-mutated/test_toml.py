def test_preserve_comments(tmp_dir):
    if False:
        while True:
            i = 10
    from dvc.utils.serialize._toml import modify_toml
    contents_fmt = '#A Title\n[foo]\nbar = {} # meaning of life\nbaz = [1, 2]\n'
    tmp_dir.gen('params.toml', contents_fmt.format('42'))
    with modify_toml('params.toml') as d:
        d['foo']['bar'] //= 2
    assert (tmp_dir / 'params.toml').read_text() == contents_fmt.format('21')

def test_parse_toml_type():
    if False:
        for i in range(10):
            print('nop')
    from tomlkit.toml_document import TOMLDocument
    from dvc.utils.serialize._toml import parse_toml
    contents = '# A Title [foo]\nbar = 42# meaning of life\nbaz = [1, 2]\n'
    parsed = parse_toml(contents, '.')
    assert not isinstance(parsed, TOMLDocument)
    assert isinstance(parsed, dict)

def test_parse_toml_for_update():
    if False:
        return 10
    from tomlkit.toml_document import TOMLDocument
    from dvc.utils.serialize._toml import parse_toml_for_update
    contents = '# A Title [foo]\nbar = 42# meaning of life\nbaz = [1, 2]\n'
    parsed = parse_toml_for_update(contents, '.')
    assert isinstance(parsed, TOMLDocument)
    assert isinstance(parsed, dict)