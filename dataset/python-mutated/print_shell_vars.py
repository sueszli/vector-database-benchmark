from spack.main import print_setup_info

def test_print_shell_vars_sh(capsys):
    if False:
        print('Hello World!')
    print_setup_info('sh')
    (out, _) = capsys.readouterr()
    assert '_sp_sys_type=' in out
    assert '_sp_tcl_roots=' in out
    assert '_sp_lmod_roots=' in out
    assert '_sp_module_prefix' not in out

def test_print_shell_vars_csh(capsys):
    if False:
        return 10
    print_setup_info('csh')
    (out, _) = capsys.readouterr()
    assert 'set _sp_sys_type = ' in out
    assert 'set _sp_tcl_roots = ' in out
    assert 'set _sp_lmod_roots = ' in out
    assert 'set _sp_module_prefix = ' not in out

def test_print_shell_vars_sh_modules(capsys):
    if False:
        print('Hello World!')
    print_setup_info('sh', 'modules')
    (out, _) = capsys.readouterr()
    assert '_sp_sys_type=' in out
    assert '_sp_tcl_roots=' in out
    assert '_sp_lmod_roots=' in out
    assert '_sp_module_prefix=' in out

def test_print_shell_vars_csh_modules(capsys):
    if False:
        for i in range(10):
            print('nop')
    print_setup_info('csh', 'modules')
    (out, _) = capsys.readouterr()
    assert 'set _sp_sys_type = ' in out
    assert 'set _sp_tcl_roots = ' in out
    assert 'set _sp_lmod_roots = ' in out
    assert 'set _sp_module_prefix = ' in out