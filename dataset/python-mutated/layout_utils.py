def assert_focused(self, name):
    if False:
        return 10
    'Asserts that window with specified name is currently focused'
    info = self.c.window.info()
    assert info['name'] == name, 'Got {0!r}, expected {1!r}'.format(info['name'], name)

def assert_dimensions(self, x, y, w, h, win=None):
    if False:
        return 10
    'Asserts dimensions of window'
    if win is None:
        win = self.c.window
    info = win.info()
    assert info['x'] == x, info
    assert info['y'] == y, info
    assert info['width'] == w, info
    assert info['height'] == h, info

def assert_dimensions_fit(self, x, y, w, h, win=None):
    if False:
        return 10
    'Asserts that window is within the given bounds'
    if win is None:
        win = self.c.window
    info = win.info()
    assert info['x'] >= x, info
    assert info['y'] >= y, info
    assert info['width'] <= w, info
    assert info['height'] <= h, info

def assert_focus_path(self, *names):
    if False:
        print('Hello World!')
    '\n    Asserts that subsequent calls to next_window() focus the open windows in\n    the given order (and prev_window() in the reverse order)\n    '
    for i in names:
        self.c.group.next_window()
        assert_focused(self, i)
    for i in names:
        self.c.group.next_window()
        assert_focused(self, i)
    for i in reversed(names):
        assert_focused(self, i)
        self.c.group.prev_window()
    for i in reversed(names):
        assert_focused(self, i)
        self.c.group.prev_window()

def assert_focus_path_unordered(self, *names):
    if False:
        for i in range(10):
            print('nop')
    '\n    Wrapper of assert_focus_path that allows the actual focus path to be\n    different from the given one, as long as:\n    1) the focus order is always the same at every forward cycle\n    2) the focus order is always the opposite at every reverse cycle\n    3) all the windows are selected once and only once at every cycle\n    '
    unordered_names = list(names)
    ordered_names = []
    while unordered_names:
        self.c.group.next_window()
        wname = self.c.window.info()['name']
        assert wname in unordered_names
        unordered_names.remove(wname)
        ordered_names.append(wname)
    assert_focus_path(ordered_names)