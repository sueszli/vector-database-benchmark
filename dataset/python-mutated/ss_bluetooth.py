import pytest
from test.widgets.test_bluetooth import dbus_thread, wait_for_text, widget

@pytest.mark.parametrize('screenshot_manager', [{}], indirect=True)
def ss_bluetooth(dbus_thread, screenshot_manager):
    if False:
        for i in range(10):
            print('nop')
    w = screenshot_manager.c.widget['bluetooth']
    wait_for_text(w, 'BT Speaker')
    screenshot_manager.take_screenshot()
    for _ in range(4):
        w.scroll_up()
        screenshot_manager.take_screenshot()