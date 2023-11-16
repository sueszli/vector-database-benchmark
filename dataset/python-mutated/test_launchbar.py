import sys
from types import ModuleType
from libqtile import widget

class MockXDG(ModuleType):

    def getIconPath(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass

def test_deprecated_configuration(caplog, monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setitem(sys.modules, 'xdg.IconTheme', MockXDG('xdg.IconTheme'))
    _ = widget.LaunchBar([('thunderbird', 'thunderbird -safe-mode', 'launch thunderbird in safe mode')])
    records = [r for r in caplog.records if r.msg.startswith('The use of')]
    assert records
    assert 'The use of a positional argument in LaunchBar is deprecated.' in records[0].msg