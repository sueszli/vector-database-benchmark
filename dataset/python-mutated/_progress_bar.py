"""Implements a progress bar widget."""
from __future__ import annotations
from math import ceil
from time import monotonic
from typing import Callable, Optional
from rich.style import Style
from .._types import UnusedParameter
from ..app import ComposeResult, RenderResult
from ..containers import Horizontal
from ..geometry import clamp
from ..reactive import reactive
from ..renderables.bar import Bar as BarRenderable
from ..timer import Timer
from ..widget import Widget
from ..widgets import Label
UNUSED = UnusedParameter()
'Sentinel for method signatures.'

class Bar(Widget, can_focus=False):
    """The bar portion of the progress bar."""
    COMPONENT_CLASSES = {'bar--bar', 'bar--complete', 'bar--indeterminate'}
    "\n    The bar sub-widget provides the component classes that follow.\n\n    These component classes let you modify the foreground and background color of the\n    bar in its different states.\n\n    | Class | Description |\n    | :- | :- |\n    | `bar--bar` | Style of the bar (may be used to change the color). |\n    | `bar--complete` | Style of the bar when it's complete. |\n    | `bar--indeterminate` | Style of the bar when it's in an indeterminate state. |\n    "
    DEFAULT_CSS = '\n    Bar {\n        width: 32;\n        height: 1;\n    }\n    Bar > .bar--bar {\n        color: $warning;\n        background: $foreground 10%;\n    }\n    Bar > .bar--indeterminate {\n        color: $error;\n        background: $foreground 10%;\n    }\n    Bar > .bar--complete {\n        color: $success;\n        background: $foreground 10%;\n    }\n    '
    _percentage: reactive[float | None] = reactive[Optional[float]](None)
    'The percentage of progress that has been completed.'
    _start_time: float | None
    'The time when the widget started tracking progress.'

    def __init__(self, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False):
        if False:
            for i in range(10):
                print('nop')
        'Create a bar for a [`ProgressBar`][textual.widgets.ProgressBar].'
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._start_time = None
        self._percentage = None

    def watch__percentage(self, percentage: float | None) -> None:
        if False:
            return 10
        'Manage the timer that enables the indeterminate bar animation.'
        if percentage is not None:
            self.auto_refresh = None
        else:
            self.auto_refresh = 1 / 15

    def render(self) -> RenderResult:
        if False:
            print('Hello World!')
        'Render the bar with the correct portion filled.'
        if self._percentage is None:
            return self.render_indeterminate()
        else:
            bar_style = self.get_component_rich_style('bar--bar') if self._percentage < 1 else self.get_component_rich_style('bar--complete')
            return BarRenderable(highlight_range=(0, self.size.width * self._percentage), highlight_style=Style.from_color(bar_style.color), background_style=Style.from_color(bar_style.bgcolor))

    def render_indeterminate(self) -> RenderResult:
        if False:
            i = 10
            return i + 15
        'Render a frame of the indeterminate progress bar animation.'
        width = self.size.width
        highlighted_bar_width = 0.25 * width
        total_imaginary_width = width + highlighted_bar_width
        speed = 30
        start = speed * self._get_elapsed_time() % (2 * total_imaginary_width)
        if start > total_imaginary_width:
            start = 2 * total_imaginary_width - start
        start -= highlighted_bar_width
        end = start + highlighted_bar_width
        bar_style = self.get_component_rich_style('bar--indeterminate')
        return BarRenderable(highlight_range=(max(0, start), min(end, width)), highlight_style=Style.from_color(bar_style.color), background_style=Style.from_color(bar_style.bgcolor))

    def _get_elapsed_time(self) -> float:
        if False:
            while True:
                i = 10
        'Get time for the indeterminate progress animation.\n\n        This method ensures that the progress bar animation always starts at the\n        beginning and it also makes it easier to test the bar if we monkey patch\n        this method.\n\n        Returns:\n            The time elapsed since the bar started being animated.\n        '
        if self._start_time is None:
            self._start_time = monotonic()
            return 0
        return monotonic() - self._start_time

class PercentageStatus(Label):
    """A label to display the percentage status of the progress bar."""
    DEFAULT_CSS = '\n    PercentageStatus {\n        width: 5;\n        content-align-horizontal: right;\n    }\n    '
    _label_text: reactive[str] = reactive('', repaint=False)
    'This is used as an auxiliary reactive to only refresh the label when needed.'
    _percentage: reactive[float | None] = reactive[Optional[float]](None)
    'The percentage of progress that has been completed.'

    def __init__(self, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False):
        if False:
            i = 10
            return i + 15
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._percentage = None
        self._label_text = '--%'

    def watch__percentage(self, percentage: float | None) -> None:
        if False:
            return 10
        'Manage the text that shows the percentage of progress.'
        if percentage is None:
            self._label_text = '--%'
        else:
            self._label_text = f'{int(100 * percentage)}%'

    def watch__label_text(self, label_text: str) -> None:
        if False:
            i = 10
            return i + 15
        'If the label text changed, update the renderable (which also refreshes).'
        self.update(label_text)

class ETAStatus(Label):
    """A label to display the estimated time until completion of the progress bar."""
    DEFAULT_CSS = '\n    ETAStatus {\n        width: 9;\n        content-align-horizontal: right;\n    }\n    '
    _label_text: reactive[str] = reactive('', repaint=False)
    'This is used as an auxiliary reactive to only refresh the label when needed.'
    _percentage: reactive[float | None] = reactive[Optional[float]](None)
    'The percentage of progress that has been completed.'
    _refresh_timer: Timer | None
    'Timer to update ETA status even when progress stalls.'
    _start_time: float | None
    'The time when the widget started tracking progress.'

    def __init__(self, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False):
        if False:
            while True:
                i = 10
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._percentage = None
        self._label_text = '--:--:--'
        self._start_time = None
        self._refresh_timer = None

    def on_mount(self) -> None:
        if False:
            print('Hello World!')
        'Periodically refresh the countdown so that the ETA is always up to date.'
        self._refresh_timer = self.set_interval(1 / 2, self.update_eta, pause=True)

    def watch__percentage(self, percentage: float | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if percentage is None:
            self._label_text = '--:--:--'
        else:
            if self._refresh_timer is not None:
                self._refresh_timer.reset()
            self.update_eta()

    def update_eta(self) -> None:
        if False:
            i = 10
            return i + 15
        'Update the ETA display.'
        percentage = self._percentage
        delta = self._get_elapsed_time()
        if not percentage or percentage >= 1 or (not delta):
            self._label_text = '--:--:--'
            if percentage is not None and percentage >= 1:
                self.auto_refresh = None
        else:
            left = ceil(delta / percentage * (1 - percentage))
            (minutes, seconds) = divmod(left, 60)
            (hours, minutes) = divmod(minutes, 60)
            if hours > 999999:
                self._label_text = '+999999h'
            elif hours > 99:
                self._label_text = f'{hours}h'
            else:
                self._label_text = f'{hours:02}:{minutes:02}:{seconds:02}'

    def _get_elapsed_time(self) -> float:
        if False:
            i = 10
            return i + 15
        'Get time to estimate time to progress completion.\n\n        Returns:\n            The time elapsed since the bar started being animated.\n        '
        if self._start_time is None:
            self._start_time = monotonic()
            return 0
        return monotonic() - self._start_time

    def watch__label_text(self, label_text: str) -> None:
        if False:
            return 10
        'If the ETA label changed, update the renderable (which also refreshes).'
        self.update(label_text)

class ProgressBar(Widget, can_focus=False):
    """A progress bar widget."""
    DEFAULT_CSS = '\n    ProgressBar > Horizontal {\n        width: auto;\n        height: auto;\n    }\n    ProgressBar {\n        width: auto;\n        height: 1;\n    }\n    '
    progress: reactive[float] = reactive(0.0)
    'The progress so far, in number of steps.'
    total: reactive[float | None] = reactive[Optional[float]](None)
    'The total number of steps associated with this progress bar, when known.\n\n    The value `None` will render an indeterminate progress bar.\n    '
    percentage: reactive[float | None] = reactive[Optional[float]](None)
    "The percentage of progress that has been completed.\n\n    The percentage is a value between 0 and 1 and the returned value is only\n    `None` if the total progress of the bar hasn't been set yet.\n\n    Example:\n        ```py\n        progress_bar = ProgressBar()\n        print(progress_bar.percentage)  # None\n        progress_bar.update(total=100)\n        progress_bar.advance(50)\n        print(progress_bar.percentage)  # 0.5\n        ```\n    "

    def __init__(self, total: float | None=None, *, show_bar: bool=True, show_percentage: bool=True, show_eta: bool=True, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False):
        if False:
            i = 10
            return i + 15
        'Create a Progress Bar widget.\n\n        The progress bar uses "steps" as the measurement unit.\n\n        Example:\n            ```py\n            class MyApp(App):\n                def compose(self):\n                    yield ProgressBar(total=100)\n\n                def key_space(self):\n                    self.query_one(ProgressBar).advance(5)\n            ```\n\n        Args:\n            total: The total number of steps in the progress if known.\n            show_bar: Whether to show the bar portion of the progress bar.\n            show_percentage: Whether to show the percentage status of the bar.\n            show_eta: Whether to show the ETA countdown of the progress bar.\n            name: The name of the widget.\n            id: The ID of the widget in the DOM.\n            classes: The CSS classes for the widget.\n            disabled: Whether the widget is disabled or not.\n        '
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.show_bar = show_bar
        self.show_percentage = show_percentage
        self.show_eta = show_eta
        self.total = total

    def compose(self) -> ComposeResult:
        if False:
            return 10

        def update_percentage(widget: Widget) -> Callable[[float | None], None]:
            if False:
                while True:
                    i = 10
            'Closure to allow updating the percentage of a given widget.'

            def updater(percentage: float | None) -> None:
                if False:
                    i = 10
                    return i + 15
                'Update the percentage reactive of the enclosed widget.'
                widget._percentage = percentage
            return updater
        with Horizontal():
            if self.show_bar:
                bar = Bar(id='bar')
                self.watch(self, 'percentage', update_percentage(bar))
                yield bar
            if self.show_percentage:
                percentage_status = PercentageStatus(id='percentage')
                self.watch(self, 'percentage', update_percentage(percentage_status))
                yield percentage_status
            if self.show_eta:
                eta_status = ETAStatus(id='eta')
                self.watch(self, 'percentage', update_percentage(eta_status))
                yield eta_status

    def validate_progress(self, progress: float) -> float:
        if False:
            while True:
                i = 10
        'Clamp the progress between 0 and the maximum total.'
        if self.total is not None:
            return clamp(progress, 0, self.total)
        return progress

    def validate_total(self, total: float | None) -> float | None:
        if False:
            return 10
        'Ensure the total is not negative.'
        if total is None:
            return total
        return max(0, total)

    def watch_total(self, total: float | None) -> None:
        if False:
            return 10
        'Re-validate progress.'
        self.progress = self.progress

    def compute_percentage(self) -> float | None:
        if False:
            i = 10
            return i + 15
        'Keep the percentage of progress updated automatically.\n\n        This will report a percentage of `1` if the total is zero.\n        '
        if self.total:
            return self.progress / self.total
        elif self.total == 0:
            return 1
        return None

    def advance(self, advance: float=1) -> None:
        if False:
            print('Hello World!')
        'Advance the progress of the progress bar by the given amount.\n\n        Example:\n            ```py\n            progress_bar.advance(10)  # Advance 10 steps.\n            ```\n\n        Args:\n            advance: Number of steps to advance progress by.\n        '
        self.progress += advance

    def update(self, *, total: None | float | UnusedParameter=UNUSED, progress: float | UnusedParameter=UNUSED, advance: float | UnusedParameter=UNUSED) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update the progress bar with the given options.\n\n        Example:\n            ```py\n            progress_bar.update(\n                total=200,  # Set new total to 200 steps.\n                progress=50,  # Set the progress to 50 (out of 200).\n            )\n            ```\n\n        Args:\n            total: New total number of steps.\n            progress: Set the progress to the given number of steps.\n            advance: Advance the progress by this number of steps.\n        '
        if not isinstance(total, UnusedParameter):
            self.total = total
        if not isinstance(progress, UnusedParameter):
            self.progress = progress
        if not isinstance(advance, UnusedParameter):
            self.progress += advance