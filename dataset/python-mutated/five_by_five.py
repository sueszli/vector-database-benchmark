"""Simple version of 5x5, developed for/with Textual."""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, cast
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.css.query import DOMQuery
from textual.reactive import reactive
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Button, Footer, Label, Markdown
if TYPE_CHECKING:
    from typing_extensions import Final

class Help(Screen):
    """The help screen for the application."""
    BINDINGS = [('escape,space,q,question_mark', 'pop_screen', 'Close')]
    'Bindings for the help screen.'

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        "Compose the game's help.\n\n        Returns:\n            ComposeResult: The result of composing the help screen.\n        "
        yield Markdown(Path(__file__).with_suffix('.md').read_text())

class WinnerMessage(Label):
    """Widget to tell the user they have won."""
    MIN_MOVES: Final = 14
    'int: The minimum number of moves you can solve the puzzle in.'

    @staticmethod
    def _plural(value: int) -> str:
        if False:
            print('Hello World!')
        return '' if value == 1 else 's'

    def show(self, moves: int) -> None:
        if False:
            while True:
                i = 10
        'Show the winner message.\n\n        Args:\n            moves (int): The number of moves required to win.\n        '
        self.update(f'W I N N E R !\n\n\nYou solved the puzzle in {moves} move{self._plural(moves)}.' + (f' It is possible to solve the puzzle in {self.MIN_MOVES}, you were {moves - self.MIN_MOVES} move{self._plural(moves - self.MIN_MOVES)} over.' if moves > self.MIN_MOVES else " Well done! That's the minimum number of moves to solve the puzzle!"))
        self.add_class('visible')

    def hide(self) -> None:
        if False:
            while True:
                i = 10
        'Hide the winner message.'
        self.remove_class('visible')

class GameHeader(Widget):
    """Header for the game.

    Comprises of the title (``#app-title``), the number of moves ``#moves``
    and the count of how many cells are turned on (``#progress``).
    """
    moves = reactive(0)
    'int: Keep track of how many moves the player has made.'
    filled = reactive(0)
    'int: Keep track of how many cells are filled.'

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        'Compose the game header.\n\n        Returns:\n            ComposeResult: The result of composing the game header.\n        '
        with Horizontal():
            yield Label(self.app.title, id='app-title')
            yield Label(id='moves')
            yield Label(id='progress')

    def watch_moves(self, moves: int):
        if False:
            print('Hello World!')
        'Watch the moves reactive and update when it changes.\n\n        Args:\n            moves (int): The number of moves made.\n        '
        self.query_one('#moves', Label).update(f'Moves: {moves}')

    def watch_filled(self, filled: int):
        if False:
            return 10
        'Watch the on-count reactive and update when it changes.\n\n        Args:\n            filled (int): The number of cells that are currently on.\n        '
        self.query_one('#progress', Label).update(f'Filled: {filled}')

class GameCell(Button):
    """Individual playable cell in the game."""

    @staticmethod
    def at(row: int, col: int) -> str:
        if False:
            while True:
                i = 10
        'Get the ID of the cell at the given location.\n\n        Args:\n            row (int): The row of the cell.\n            col (int): The column of the cell.\n\n        Returns:\n            str: A string ID for the cell.\n        '
        return f'cell-{row}-{col}'

    def __init__(self, row: int, col: int) -> None:
        if False:
            print('Hello World!')
        'Initialise the game cell.\n\n        Args:\n            row (int): The row of the cell.\n            col (int): The column of the cell.\n        '
        super().__init__('', id=self.at(row, col))
        self.row = row
        self.col = col

class GameGrid(Widget):
    """The main playable grid of game cells."""

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        'Compose the game grid.\n\n        Returns:\n            ComposeResult: The result of composing the game grid.\n        '
        for row in range(Game.SIZE):
            for col in range(Game.SIZE):
                yield GameCell(row, col)

class Game(Screen):
    """Main 5x5 game grid screen."""
    SIZE: Final = 5
    "The size of the game grid. Clue's in the name really."
    BINDINGS = [Binding('n', 'new_game', 'New Game'), Binding('question_mark', "push_screen('help')", 'Help', key_display='?'), Binding('q', 'quit', 'Quit'), Binding('up,w,k', 'navigate(-1,0)', 'Move Up', False), Binding('down,s,j', 'navigate(1,0)', 'Move Down', False), Binding('left,a,h', 'navigate(0,-1)', 'Move Left', False), Binding('right,d,l', 'navigate(0,1)', 'Move Right', False), Binding('space', 'move', 'Toggle', False)]
    'The bindings for the main game grid.'

    @property
    def filled_cells(self) -> DOMQuery[GameCell]:
        if False:
            print('Hello World!')
        'DOMQuery[GameCell]: The collection of cells that are currently turned on.'
        return cast(DOMQuery[GameCell], self.query('GameCell.filled'))

    @property
    def filled_count(self) -> int:
        if False:
            while True:
                i = 10
        'int: The number of cells that are currently filled.'
        return len(self.filled_cells)

    @property
    def all_filled(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'bool: Are all the cells filled?'
        return self.filled_count == self.SIZE * self.SIZE

    def game_playable(self, playable: bool) -> None:
        if False:
            return 10
        'Mark the game as playable, or not.\n\n        Args:\n            playable (bool): Should the game currently be playable?\n        '
        self.query_one(GameGrid).disabled = not playable

    def cell(self, row: int, col: int) -> GameCell:
        if False:
            i = 10
            return i + 15
        'Get the cell at a given location.\n\n        Args:\n            row (int): The row of the cell to get.\n            col (int): The column of the cell to get.\n\n        Returns:\n            GameCell: The cell at that location.\n        '
        return self.query_one(f'#{GameCell.at(row, col)}', GameCell)

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        'Compose the game screen.\n\n        Returns:\n            ComposeResult: The result of composing the game screen.\n        '
        yield GameHeader()
        yield GameGrid()
        yield Footer()
        yield WinnerMessage()

    def toggle_cell(self, row: int, col: int) -> None:
        if False:
            i = 10
            return i + 15
        "Toggle an individual cell, but only if it's in bounds.\n\n        If the row and column would place the cell out of bounds for the\n        game grid, this function call is a no-op. That is, it's safe to call\n        it with an invalid cell coordinate.\n\n        Args:\n            row (int): The row of the cell to toggle.\n            col (int): The column of the cell to toggle.\n        "
        if 0 <= row <= self.SIZE - 1 and 0 <= col <= self.SIZE - 1:
            self.cell(row, col).toggle_class('filled')
    _PATTERN: Final = (-1, 1, 0, 0, 0)

    def toggle_cells(self, cell: GameCell) -> None:
        if False:
            print('Hello World!')
        'Toggle a 5x5 pattern around the given cell.\n\n        Args:\n            cell (GameCell): The cell to toggle the cells around.\n        '
        for (row, col) in zip(self._PATTERN, reversed(self._PATTERN)):
            self.toggle_cell(cell.row + row, cell.col + col)
        self.query_one(GameHeader).filled = self.filled_count

    def make_move_on(self, cell: GameCell) -> None:
        if False:
            i = 10
            return i + 15
        "Make a move on the given cell.\n\n        All relevant cells around the given cell are toggled as per the\n        game's rules.\n\n        Args:\n            cell (GameCell): The cell to make a move on\n        "
        self.toggle_cells(cell)
        self.query_one(GameHeader).moves += 1
        if self.all_filled:
            self.query_one(WinnerMessage).show(self.query_one(GameHeader).moves)
            self.game_playable(False)

    def on_button_pressed(self, event: GameCell.Pressed) -> None:
        if False:
            i = 10
            return i + 15
        'React to a press of a button on the game grid.\n\n        Args:\n            event (GameCell.Pressed): The event to react to.\n        '
        self.make_move_on(cast(GameCell, event.button))

    def action_new_game(self) -> None:
        if False:
            i = 10
            return i + 15
        'Start a new game.'
        self.query_one(GameHeader).moves = 0
        self.filled_cells.remove_class('filled')
        self.query_one(WinnerMessage).hide()
        middle = self.cell(self.SIZE // 2, self.SIZE // 2)
        self.toggle_cells(middle)
        self.set_focus(middle)
        self.game_playable(True)

    def action_navigate(self, row: int, col: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Navigate to a new cell by the given offsets.\n\n        Args:\n            row (int): The row of the cell to navigate to.\n            col (int): The column of the cell to navigate to.\n        '
        if isinstance(self.focused, GameCell):
            self.set_focus(self.cell((self.focused.row + row) % self.SIZE, (self.focused.col + col) % self.SIZE))

    def action_move(self) -> None:
        if False:
            while True:
                i = 10
        'Make a move on the current cell.'
        if isinstance(self.focused, GameCell):
            self.focused.press()

    def on_mount(self) -> None:
        if False:
            return 10
        'Get the game started when we first mount.'
        self.action_new_game()

class FiveByFive(App[None]):
    """Main 5x5 application class."""
    CSS_PATH = 'five_by_five.tcss'
    'The name of the stylesheet for the app.'
    SCREENS = {'help': Help}
    'The pre-loaded screens for the application.'
    BINDINGS = [('ctrl+d', 'toggle_dark', 'Toggle Dark Mode')]
    'App-level bindings.'
    TITLE = '5x5 -- A little annoying puzzle'
    'The title of the application.'

    def on_mount(self) -> None:
        if False:
            print('Hello World!')
        'Set up the application on startup.'
        self.push_screen(Game())
if __name__ == '__main__':
    FiveByFive().run()