"""Commands related to caret browsing."""
from qutebrowser.api import cmdutils, apitypes

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
@cmdutils.argument('count', value=cmdutils.Value.count)
def move_to_next_line(tab: apitypes.Tab, count: int=1) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Move the cursor or selection to the next line.\n\n    Args:\n        count: How many lines to move.\n    '
    tab.caret.move_to_next_line(count)

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
@cmdutils.argument('count', value=cmdutils.Value.count)
def move_to_prev_line(tab: apitypes.Tab, count: int=1) -> None:
    if False:
        print('Hello World!')
    'Move the cursor or selection to the prev line.\n\n    Args:\n        count: How many lines to move.\n    '
    tab.caret.move_to_prev_line(count)

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
@cmdutils.argument('count', value=cmdutils.Value.count)
def move_to_next_char(tab: apitypes.Tab, count: int=1) -> None:
    if False:
        return 10
    'Move the cursor or selection to the next char.\n\n    Args:\n        count: How many lines to move.\n    '
    tab.caret.move_to_next_char(count)

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
@cmdutils.argument('count', value=cmdutils.Value.count)
def move_to_prev_char(tab: apitypes.Tab, count: int=1) -> None:
    if False:
        return 10
    'Move the cursor or selection to the previous char.\n\n    Args:\n        count: How many chars to move.\n    '
    tab.caret.move_to_prev_char(count)

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
@cmdutils.argument('count', value=cmdutils.Value.count)
def move_to_end_of_word(tab: apitypes.Tab, count: int=1) -> None:
    if False:
        return 10
    'Move the cursor or selection to the end of the word.\n\n    Args:\n        count: How many words to move.\n    '
    tab.caret.move_to_end_of_word(count)

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
@cmdutils.argument('count', value=cmdutils.Value.count)
def move_to_next_word(tab: apitypes.Tab, count: int=1) -> None:
    if False:
        i = 10
        return i + 15
    'Move the cursor or selection to the next word.\n\n    Args:\n        count: How many words to move.\n    '
    tab.caret.move_to_next_word(count)

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
@cmdutils.argument('count', value=cmdutils.Value.count)
def move_to_prev_word(tab: apitypes.Tab, count: int=1) -> None:
    if False:
        return 10
    'Move the cursor or selection to the previous word.\n\n    Args:\n        count: How many words to move.\n    '
    tab.caret.move_to_prev_word(count)

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
def move_to_start_of_line(tab: apitypes.Tab) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Move the cursor or selection to the start of the line.'
    tab.caret.move_to_start_of_line()

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
def move_to_end_of_line(tab: apitypes.Tab) -> None:
    if False:
        print('Hello World!')
    'Move the cursor or selection to the end of line.'
    tab.caret.move_to_end_of_line()

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
@cmdutils.argument('count', value=cmdutils.Value.count)
def move_to_start_of_next_block(tab: apitypes.Tab, count: int=1) -> None:
    if False:
        while True:
            i = 10
    'Move the cursor or selection to the start of next block.\n\n    Args:\n        count: How many blocks to move.\n    '
    tab.caret.move_to_start_of_next_block(count)

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
@cmdutils.argument('count', value=cmdutils.Value.count)
def move_to_start_of_prev_block(tab: apitypes.Tab, count: int=1) -> None:
    if False:
        while True:
            i = 10
    'Move the cursor or selection to the start of previous block.\n\n    Args:\n        count: How many blocks to move.\n    '
    tab.caret.move_to_start_of_prev_block(count)

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
@cmdutils.argument('count', value=cmdutils.Value.count)
def move_to_end_of_next_block(tab: apitypes.Tab, count: int=1) -> None:
    if False:
        while True:
            i = 10
    'Move the cursor or selection to the end of next block.\n\n    Args:\n        count: How many blocks to move.\n    '
    tab.caret.move_to_end_of_next_block(count)

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
@cmdutils.argument('count', value=cmdutils.Value.count)
def move_to_end_of_prev_block(tab: apitypes.Tab, count: int=1) -> None:
    if False:
        return 10
    'Move the cursor or selection to the end of previous block.\n\n    Args:\n        count: How many blocks to move.\n    '
    tab.caret.move_to_end_of_prev_block(count)

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
def move_to_start_of_document(tab: apitypes.Tab) -> None:
    if False:
        while True:
            i = 10
    'Move the cursor or selection to the start of the document.'
    tab.caret.move_to_start_of_document()

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
def move_to_end_of_document(tab: apitypes.Tab) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Move the cursor or selection to the end of the document.'
    tab.caret.move_to_end_of_document()

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
def selection_toggle(tab: apitypes.Tab, line: bool=False) -> None:
    if False:
        return 10
    'Toggle caret selection mode.\n\n    Args:\n        line: Enables line-selection.\n    '
    tab.caret.toggle_selection(line)

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
def selection_drop(tab: apitypes.Tab) -> None:
    if False:
        while True:
            i = 10
    'Drop selection and keep selection mode enabled.'
    tab.caret.drop_selection()

@cmdutils.register()
@cmdutils.argument('tab_obj', value=cmdutils.Value.cur_tab)
def selection_follow(tab_obj: apitypes.Tab, *, tab: bool=False) -> None:
    if False:
        print('Hello World!')
    'Follow the selected text.\n\n    Args:\n        tab: Load the selected link in a new tab.\n    '
    try:
        tab_obj.caret.follow_selected(tab=tab)
    except apitypes.WebTabError as e:
        raise cmdutils.CommandError(str(e))

@cmdutils.register(modes=[cmdutils.KeyMode.caret])
@cmdutils.argument('tab', value=cmdutils.Value.cur_tab)
def selection_reverse(tab: apitypes.Tab) -> None:
    if False:
        print('Hello World!')
    'Swap the stationary and moving end of the current selection.'
    tab.caret.reverse_selection()