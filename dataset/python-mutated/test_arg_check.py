import contextlib
import io
from unittest import mock
import pytest
from mitmproxy.utils import arg_check

@pytest.mark.parametrize('arg, output', [(['-T'], '-T is deprecated, please use --mode transparent instead'), (['-U'], '-U is deprecated, please use --mode upstream:SPEC instead'), (['--confdir'], '--confdir is deprecated.\nPlease use `--set confdir=value` instead.\nTo show all options and their default values use --options'), (['--palette'], '--palette is deprecated.\nPlease use `--set console_palette=value` instead.\nTo show all options and their default values use --options'), (['--wfile'], '--wfile is deprecated.\nPlease use `--save-stream-file` instead.'), (['--eventlog'], '--eventlog has been removed.'), (['--nonanonymous'], '--nonanonymous is deprecated.\nPlease use `--proxyauth SPEC` instead.\nSPEC Format: "username:pass", "any" to accept any user/pass combination,\n"@path" to use an Apache htpasswd file, or\n"ldap[s]:url_server_ldap[:port]:dn_auth:password:dn_subtree[?search_filter_key=...]" for LDAP authentication.'), (['--replacements'], '--replacements is deprecated.\nPlease use `--modify-body` or `--modify-headers` instead.'), (['--underscore_option'], '--underscore_option uses underscores, please use hyphens --underscore-option')])
def test_check_args(arg, output):
    if False:
        for i in range(10):
            print('nop')
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        with mock.patch('sys.argv') as m:
            m.__getitem__.return_value = arg
            arg_check.check()
            assert f.getvalue().strip() == output