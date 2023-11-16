import pytest
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.core_test]

def test_jinja_renderer_argline(state, state_tree):
    if False:
        return 10
    '\n    This is a test case for https://github.com/saltstack/salt/issues/55124\n    '
    renderer_contents = '\n\n    import salt.utils.stringio\n\n\n    def render(gpg_data, saltenv="base", sls="", argline="", **kwargs):\n        \'\'\'\n        Renderer which returns the text value of the SLS file, instead of a\n        StringIO object.\n        \'\'\'\n        if salt.utils.stringio.is_readable(gpg_data):\n            return gpg_data.getvalue()\n        else:\n            return gpg_data\n    '
    sls_contents = "\n    #!issue55124|jinja -s|yaml\n\n    'Who am I?':\n      cmd.run:\n        - name: echo {{ salt.cmd.run('whoami') }}\n    "
    with pytest.helpers.temp_file('issue51499.py', renderer_contents, state_tree / '_renderers'), pytest.helpers.temp_file('issue-55124.sls', sls_contents, state_tree):
        ret = state.sls('issue-55124')
        for state_return in ret:
            assert state_return.result is True