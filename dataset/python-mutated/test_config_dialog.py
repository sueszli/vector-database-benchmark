"""
Tests for configdialog.py
"""

def test_config_dialog_save_to_conf(global_config_dialog):
    if False:
        print('Hello World!')
    for index in range(global_config_dialog.pages_widget.count()):
        configpage = global_config_dialog.get_page(index)
        configpage.save_to_conf()
        _save_lang = getattr(configpage, '_save_lang', None)
        if _save_lang:
            _save_lang()
        assert configpage.is_valid()