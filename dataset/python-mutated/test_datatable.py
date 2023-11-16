import flet_core as ft
import pytest
from flet_core.protocol import Command

def test_datatable_instance_no_attrs_set():
    if False:
        print('Hello World!')
    r = ft.DataTable()
    assert isinstance(r, ft.Control)
    assert r._build_add_commands() == [Command(indent=0, name=None, values=['datatable'], attrs={}, commands=[])], 'Test failed'

def test_datarow_instance_no_attrs_set():
    if False:
        i = 10
        return i + 15
    r = ft.DataRow()
    assert isinstance(r, ft.Control)
    assert r._build_add_commands() == [Command(indent=0, name=None, values=['r'], attrs={}, commands=[])], 'Test failed'

def test_datarow_color_literal_material_state_as_string():
    if False:
        for i in range(10):
            print('nop')
    r = ft.DataRow(color='yellow')
    assert isinstance(r, ft.Control)
    assert r._build_add_commands() == [Command(indent=0, name=None, values=['r'], attrs={'color': '"yellow"'}, commands=[])], 'Test failed'

def test_datarow_color_multiple_material_states_as_strings():
    if False:
        i = 10
        return i + 15
    r = ft.DataRow(color={'selected': 'red', 'hovered': 'blue', '': 'yellow'})
    assert isinstance(r, ft.Control)
    assert r._build_add_commands() == [Command(indent=0, name=None, values=['r'], attrs={'color': '{"selected":"red","hovered":"blue","":"yellow"}'}, commands=[])], 'Test failed'

def test_datarow_color_multiple_material_states():
    if False:
        i = 10
        return i + 15
    r = ft.DataRow(color={ft.MaterialState.SELECTED: 'red', ft.MaterialState.HOVERED: 'blue', ft.MaterialState.DEFAULT: 'yellow'})
    assert isinstance(r, ft.Control)
    assert r._build_add_commands() == [Command(indent=0, name=None, values=['r'], attrs={'color': '{"selected":"red","hovered":"blue","":"yellow"}'}, commands=[])], 'Test failed'