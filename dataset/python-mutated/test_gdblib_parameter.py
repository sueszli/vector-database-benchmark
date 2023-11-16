from __future__ import annotations
import gdb
import pytest
import pwndbg.gdblib.config

@pytest.mark.parametrize('params', (('int', 123, '123', {}), ('bool', True, 'on', {}), ('bool', False, 'off', {}), ('string', 'some-string-val', 'some-string-val', {}), ('auto-bool', None, 'auto', {'param_class': gdb.PARAM_AUTO_BOOLEAN}), ('unlimited-uint', 0, 'unlimited', {'param_class': gdb.PARAM_UINTEGER}), ('unlimited-int', 0, 'unlimited', {'param_class': gdb.PARAM_INTEGER}), ('enum', 'enum1', 'enum1', {'param_class': gdb.PARAM_ENUM, 'enum_sequence': ['enum1', 'enum2', 'enum3']}), ('zuint', 0, '0', {'param_class': gdb.PARAM_ZUINTEGER if hasattr(gdb, 'PARAM_ZUINTEGER') else 'PARAM_ZUINTEGER'}), ('unlimited-zuint', -1, 'unlimited', {'param_class': gdb.PARAM_ZUINTEGER_UNLIMITED if hasattr(gdb, 'PARAM_ZUINTEGER_UNLIMITED') else 'PARAM_ZUINTEGER_UNLIMITED'})))
def test_gdb_parameter_default_value_works(start_binary, params):
    if False:
        for i in range(10):
            print('nop')
    if not params:
        pytest.skip('Current GDB version does not support this testcase')
    (name_suffix, default_value, displayed_value, optional_kwargs) = params
    param_name = f'test-param-{name_suffix}'
    help_docstring = f'Help docstring for {param_name}'
    set_show_doc = 'the value of the foo'
    param = pwndbg.gdblib.config.add_param(param_name, default_value, set_show_doc, help_docstring=help_docstring, **optional_kwargs)
    pwndbg.gdblib.config_mod.Parameter(param)
    out = gdb.execute(f'show {param_name}', to_string=True)
    assert out == f'{set_show_doc.capitalize()} is {displayed_value!r}. See `help set {param_name}` for more information.\n'
    if optional_kwargs.get('param_class') in (gdb.PARAM_UINTEGER, gdb.PARAM_INTEGER) and default_value == 0:
        assert gdb.parameter(param_name) is None
    else:
        assert gdb.parameter(param_name) == default_value
    assert param.value == default_value
    out = gdb.execute(f'help show {param_name}', to_string=True)
    assert out == f'Show {set_show_doc}.\n{help_docstring}\n'
    assert gdb.execute(f'help set {param_name}', to_string=True) == f'Set {set_show_doc}.\n{help_docstring}\n'