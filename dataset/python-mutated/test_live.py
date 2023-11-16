import time
from typing import Optional
from rich.console import Console
from rich.live import Live
from rich.text import Text

def create_capture_console(*, width: int=60, height: int=80, force_terminal: Optional[bool]=True) -> Console:
    if False:
        while True:
            i = 10
    return Console(width=width, height=height, force_terminal=force_terminal, legacy_windows=False, color_system=None, _environ={})

def test_live_state() -> None:
    if False:
        i = 10
        return i + 15
    with Live('') as live:
        assert live._started
        live.start()
        assert live.renderable == ''
        assert live._started
        live.stop()
        assert not live._started
    assert not live._started

def test_growing_display() -> None:
    if False:
        print('Hello World!')
    console = create_capture_console()
    console.begin_capture()
    with Live(console=console, auto_refresh=False) as live:
        display = ''
        for step in range(10):
            display += f'Step {step}\n'
            live.update(display, refresh=True)
    output = console.end_capture()
    print(repr(output))
    assert output == '\x1b[?25lStep 0\n\r\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\nStep 9\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\nStep 9\n\n\x1b[?25h'

def test_growing_display_transient() -> None:
    if False:
        return 10
    console = create_capture_console()
    console.begin_capture()
    with Live(console=console, auto_refresh=False, transient=True) as live:
        display = ''
        for step in range(10):
            display += f'Step {step}\n'
            live.update(display, refresh=True)
    output = console.end_capture()
    assert output == '\x1b[?25lStep 0\n\r\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\nStep 9\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\nStep 9\n\n\x1b[?25h\r\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K'

def test_growing_display_overflow_ellipsis() -> None:
    if False:
        while True:
            i = 10
    console = create_capture_console(height=5)
    console.begin_capture()
    with Live(console=console, auto_refresh=False, vertical_overflow='ellipsis') as live:
        display = ''
        for step in range(10):
            display += f'Step {step}\n'
            live.update(display, refresh=True)
    output = console.end_capture()
    assert output == '\x1b[?25lStep 0\n\r\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\n                            ...                             \r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\n                            ...                             \r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\n                            ...                             \r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\n                            ...                             \r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\n                            ...                             \r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\n                            ...                             \r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\nStep 9\n\n\x1b[?25h'

def test_growing_display_overflow_crop() -> None:
    if False:
        print('Hello World!')
    console = create_capture_console(height=5)
    console.begin_capture()
    with Live(console=console, auto_refresh=False, vertical_overflow='crop') as live:
        display = ''
        for step in range(10):
            display += f'Step {step}\n'
            live.update(display, refresh=True)
    output = console.end_capture()
    assert output == '\x1b[?25lStep 0\n\r\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\nStep 9\n\n\x1b[?25h'

def test_growing_display_overflow_visible() -> None:
    if False:
        return 10
    console = create_capture_console(height=5)
    console.begin_capture()
    with Live(console=console, auto_refresh=False, vertical_overflow='visible') as live:
        display = ''
        for step in range(10):
            display += f'Step {step}\n'
            live.update(display, refresh=True)
    output = console.end_capture()
    assert output == '\x1b[?25lStep 0\n\r\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\nStep 9\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\nStep 9\n\n\x1b[?25h'

def test_growing_display_autorefresh() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Test generating a table but using auto-refresh from threading'
    console = create_capture_console(height=5)
    console.begin_capture()
    with Live(console=console, auto_refresh=True, vertical_overflow='visible') as live:
        display = ''
        for step in range(10):
            display += f'Step {step}\n'
            live.update(display)
            time.sleep(0.2)

def test_growing_display_console_redirect() -> None:
    if False:
        while True:
            i = 10
    console = create_capture_console()
    console.begin_capture()
    with Live(console=console, auto_refresh=False) as live:
        display = ''
        for step in range(10):
            console.print(f'Running step {step}')
            display += f'Step {step}\n'
            live.update(display, refresh=True)
    output = console.end_capture()
    assert output == '\x1b[?25lRunning step 0\n\r\x1b[2KStep 0\n\r\x1b[2K\x1b[1A\x1b[2KRunning step 1\nStep 0\n\r\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KRunning step 2\nStep 0\nStep 1\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KRunning step 3\nStep 0\nStep 1\nStep 2\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KRunning step 4\nStep 0\nStep 1\nStep 2\nStep 3\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KRunning step 5\nStep 0\nStep 1\nStep 2\nStep 3\nStep 4\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KRunning step 6\nStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KRunning step 7\nStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KRunning step 8\nStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KRunning step 9\nStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\nStep 9\n\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2KStep 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\nStep 9\n\n\x1b[?25h'

def test_growing_display_file_console() -> None:
    if False:
        return 10
    console = create_capture_console(force_terminal=False)
    console.begin_capture()
    with Live(console=console, auto_refresh=False) as live:
        display = ''
        for step in range(10):
            display += f'Step {step}\n'
            live.update(display, refresh=True)
    output = console.end_capture()
    assert output == 'Step 0\nStep 1\nStep 2\nStep 3\nStep 4\nStep 5\nStep 6\nStep 7\nStep 8\nStep 9\n'

def test_live_screen() -> None:
    if False:
        return 10
    console = create_capture_console(width=20, height=5)
    console.begin_capture()
    with Live(Text('foo'), screen=True, console=console, auto_refresh=False) as live:
        live.refresh()
    result = console.end_capture()
    print(repr(result))
    expected = '\x1b[?1049h\x1b[H\x1b[?25l\x1b[Hfoo                 \n                    \n                    \n                    \n                    \x1b[Hfoo                 \n                    \n                    \n                    \n                    \x1b[?25h\x1b[?1049l'
    assert result == expected