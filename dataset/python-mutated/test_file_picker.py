import flet_core as ft
import pytest
from flet_core.protocol import Command

def test_instance_no_attrs_set():
    if False:
        i = 10
        return i + 15
    r = ft.FilePicker()
    assert isinstance(r, ft.Control)
    assert r._build_add_commands() == [Command(indent=0, name=None, values=['filepicker'], attrs={'upload': '[]'}, commands=[])], 'Test failed'

def test_file_type_enum():
    if False:
        return 10
    r = ft.FilePicker()
    r.file_type = ft.FilePickerFileType.VIDEO
    assert isinstance(r.file_type, ft.FilePickerFileType)
    assert r.file_type == ft.FilePickerFileType.VIDEO
    assert r._get_attr('fileType') == 'video'
    r = ft.FilePicker()
    r.file_type = 'any'
    assert isinstance(r.file_type, str)
    assert r._get_attr('fileType') == 'any'