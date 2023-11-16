from textwrap import dedent
from mock import patch
from nameko.cli.commands import ShowConfig
from nameko.cli.main import setup_parser, setup_yaml_parser

@patch('nameko.cli.main.os')
def test_main(mock_os, tmpdir, capsys):
    if False:
        for i in range(10):
            print('nop')
    config = tmpdir.join('config.yaml')
    config.write('\n        FOO: ${FOO:foobar}\n        BAR: ${BAR}\n    ')
    parser = setup_parser()
    setup_yaml_parser()
    args = parser.parse_args(['show-config', '--config', config.strpath])
    mock_os.environ = {'BAR': '[1,2,3]'}
    ShowConfig.main(args)
    (out, _) = capsys.readouterr()
    expected = dedent('\n        BAR:\n        - 1\n        - 2\n        - 3\n        FOO: foobar\n    ').strip()
    assert out.strip() == expected