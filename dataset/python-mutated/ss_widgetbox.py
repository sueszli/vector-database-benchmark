import pytest
import libqtile.widget

@pytest.fixture
def widget(monkeypatch):
    if False:
        return 10

    class WidgetBox(libqtile.widget.WidgetBox):

        def _configure(self, bar, screen):
            if False:
                print('Hello World!')
            libqtile.widget.WidgetBox._configure(self, bar, screen)
            for w in self.widgets:
                w.drawer.has_mirrors = True
    yield WidgetBox

@pytest.mark.parametrize('screenshot_manager', [{'widgets': [libqtile.widget.TextBox('Widget inside box.')]}], indirect=True)
def ss_widgetbox(screenshot_manager):
    if False:
        return 10
    bar = screenshot_manager.c.bar['top']

    def bar_width():
        if False:
            i = 10
            return i + 15
        info = bar.info()
        widgets = info['widgets']
        if not widgets:
            return 0
        return sum((x['length'] for x in widgets))

    def take_screenshot():
        if False:
            print('Hello World!')
        target = screenshot_manager.target()
        bar.take_screenshot(target, width=bar_width())
    take_screenshot()
    screenshot_manager.c.widget['widgetbox'].toggle()
    take_screenshot()