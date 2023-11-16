from __future__ import annotations
import pytest
from textual.app import App, ComposeResult
from textual.geometry import Offset
from textual.widgets import Input
TEXT_SINGLE = 'That gum you like is going to come back in style'
TEXT_DOUBLE = 'こんにちは'
TEXT_MIXED = 'aこんbcにdちeは'

class InputApp(App[None]):

    def __init__(self, text):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield Input(self._text)

@pytest.mark.parametrize('text, click_at, should_land', ((TEXT_SINGLE, 0, 0), (TEXT_SINGLE, 1, 1), (TEXT_SINGLE, 10, 10), (TEXT_SINGLE, len(TEXT_SINGLE) - 1, len(TEXT_SINGLE) - 1), (TEXT_SINGLE, len(TEXT_SINGLE), len(TEXT_SINGLE)), (TEXT_SINGLE, len(TEXT_SINGLE) + 10, len(TEXT_SINGLE)), (TEXT_DOUBLE, 0, 0), (TEXT_DOUBLE, 1, 0), (TEXT_DOUBLE, 2, 1), (TEXT_DOUBLE, 3, 1), (TEXT_DOUBLE, 4, 2), (TEXT_DOUBLE, 5, 2), (TEXT_DOUBLE, len(TEXT_DOUBLE) * 2 - 1, len(TEXT_DOUBLE) - 1), (TEXT_DOUBLE, len(TEXT_DOUBLE) * 2, len(TEXT_DOUBLE)), (TEXT_DOUBLE, len(TEXT_DOUBLE) * 10, len(TEXT_DOUBLE)), (TEXT_MIXED, 0, 0), (TEXT_MIXED, 1, 1), (TEXT_MIXED, 2, 1), (TEXT_MIXED, 3, 2), (TEXT_MIXED, 4, 2), (TEXT_MIXED, 5, 3), (TEXT_MIXED, 13, 9), (TEXT_MIXED, 14, 9), (TEXT_MIXED, 15, 10), (TEXT_MIXED, 60, 10)))
async def test_mouse_clicks_within(text, click_at, should_land):
    """Mouse clicks should result in the cursor going to the right place."""
    async with InputApp(text).run_test() as pilot:
        await pilot.click(Input, Offset(click_at + 3, 1))
        await pilot.pause()
        assert pilot.app.query_one(Input).cursor_position == should_land

async def test_mouse_click_outwith():
    """Mouse clicks outside the input should not affect cursor position."""
    async with InputApp(TEXT_SINGLE).run_test() as pilot:
        pilot.app.query_one(Input).cursor_position = 3
        assert pilot.app.query_one(Input).cursor_position == 3
        await pilot.click(Input, Offset(0, 0))
        await pilot.pause()
        assert pilot.app.query_one(Input).cursor_position == 3