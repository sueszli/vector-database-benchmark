import textwrap

def test_popup_focus(manager):
    if False:
        i = 10
        return i + 15
    manager.test_window('one')
    start_wins = len(manager.backend.get_all_windows())
    (success, msg) = manager.c.eval(textwrap.dedent('\n        from libqtile.popup import Popup\n        popup = Popup(self,\n            x=0,\n            y=0,\n            width=self.current_screen.width,\n            height=self.current_screen.height,\n        )\n        popup.place()\n        popup.unhide()\n    '))
    assert success, msg
    end_wins = len(manager.backend.get_all_windows())
    assert end_wins == start_wins + 1
    assert manager.c.group.info()['focus'] == 'one'
    assert manager.c.group.info()['windows'] == ['one']
    assert len(manager.c.windows()) == 1