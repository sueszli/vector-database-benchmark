from rich._palettes import STANDARD_PALETTE
from rich.table import Table

def test_rich_cast():
    if False:
        i = 10
        return i + 15
    table = STANDARD_PALETTE.__rich__()
    assert isinstance(table, Table)
    assert table.row_count == 16