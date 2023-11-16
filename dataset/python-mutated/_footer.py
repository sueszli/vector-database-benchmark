from __future__ import annotations
from collections import defaultdict
from typing import ClassVar, Optional
import rich.repr
from rich.console import RenderableType
from rich.text import Text
from .. import events
from ..reactive import reactive
from ..widget import Widget

@rich.repr.auto
class Footer(Widget):
    """A simple footer widget which docks itself to the bottom of the parent container."""
    COMPONENT_CLASSES: ClassVar[set[str]] = {'footer--description', 'footer--key', 'footer--highlight', 'footer--highlight-key'}
    '\n    | Class | Description |\n    | :- | :- |\n    | `footer--description` | Targets the descriptions of the key bindings. |\n    | `footer--highlight` | Targets the highlighted key binding. |\n    | `footer--highlight-key` | Targets the key portion of the highlighted key binding. |\n    | `footer--key` | Targets the key portions of the key bindings. |\n    '
    DEFAULT_CSS = '\n    Footer {\n        background: $accent;\n        color: $text;\n        dock: bottom;\n        height: 1;\n    }\n    Footer > .footer--highlight {\n        background: $accent-darken-1;\n    }\n\n    Footer > .footer--highlight-key {\n        background: $secondary;\n        text-style: bold;\n    }\n\n    Footer > .footer--key {\n        text-style: bold;\n        background: $accent-darken-2;\n    }\n    '
    highlight_key: reactive[str | None] = reactive[Optional[str]](None)

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self._key_text: Text | None = None
        self.auto_links = False

    async def watch_highlight_key(self) -> None:
        """If highlight key changes we need to regenerate the text."""
        self._key_text = None
        self.refresh()

    def _on_mount(self, _: events.Mount) -> None:
        if False:
            print('Hello World!')
        self.watch(self.screen, 'focused', self._bindings_changed)
        self.watch(self.screen, 'stack_updates', self._bindings_changed)

    def _bindings_changed(self, _: Widget | None) -> None:
        if False:
            while True:
                i = 10
        self._key_text = None
        self.refresh()

    def _on_mouse_move(self, event: events.MouseMove) -> None:
        if False:
            return 10
        'Store any key we are moving over.'
        self.highlight_key = event.style.meta.get('key')

    def _on_leave(self, _: events.Leave) -> None:
        if False:
            while True:
                i = 10
        'Clear any highlight when the mouse leaves the widget'
        self.highlight_key = None

    def __rich_repr__(self) -> rich.repr.Result:
        if False:
            return 10
        yield from super().__rich_repr__()

    def _make_key_text(self) -> Text:
        if False:
            return 10
        'Create text containing all the keys.'
        base_style = self.rich_style
        text = Text(style=self.rich_style, no_wrap=True, overflow='ellipsis', justify='left', end='')
        highlight_style = self.get_component_rich_style('footer--highlight')
        highlight_key_style = self.get_component_rich_style('footer--highlight-key')
        key_style = self.get_component_rich_style('footer--key')
        description_style = self.get_component_rich_style('footer--description')
        bindings = [binding for (_, binding) in self.app.namespace_bindings.values() if binding.show]
        action_to_bindings = defaultdict(list)
        for binding in bindings:
            action_to_bindings[binding.action].append(binding)
        for (_, bindings) in action_to_bindings.items():
            binding = bindings[0]
            if binding.key_display is None:
                key_display = self.app.get_key_display(binding.key)
                if key_display is None:
                    key_display = binding.key.upper()
            else:
                key_display = binding.key_display
            hovered = self.highlight_key == binding.key
            key_text = Text.assemble((f' {key_display} ', highlight_key_style if hovered else key_style), (f' {binding.description} ', highlight_style if hovered else base_style + description_style), meta={'@click': f"app.check_bindings('{binding.key}')", 'key': binding.key})
            text.append_text(key_text)
        return text

    def notify_style_update(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._key_text = None

    def post_render(self, renderable):
        if False:
            while True:
                i = 10
        return renderable

    def render(self) -> RenderableType:
        if False:
            i = 10
            return i + 15
        if self._key_text is None:
            self._key_text = self._make_key_text()
        return self._key_text