from __future__ import annotations
from typing import TYPE_CHECKING
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.core.shared_docs import _shared_docs
from pandas.io.excel._base import BaseExcelReader
if TYPE_CHECKING:
    from pyxlsb import Workbook
    from pandas._typing import FilePath, ReadBuffer, Scalar, StorageOptions

class PyxlsbReader(BaseExcelReader['Workbook']):

    @doc(storage_options=_shared_docs['storage_options'])
    def __init__(self, filepath_or_buffer: FilePath | ReadBuffer[bytes], storage_options: StorageOptions | None=None, engine_kwargs: dict | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Reader using pyxlsb engine.\n\n        Parameters\n        ----------\n        filepath_or_buffer : str, path object, or Workbook\n            Object to be parsed.\n        {storage_options}\n        engine_kwargs : dict, optional\n            Arbitrary keyword arguments passed to excel engine.\n        '
        import_optional_dependency('pyxlsb')
        super().__init__(filepath_or_buffer, storage_options=storage_options, engine_kwargs=engine_kwargs)

    @property
    def _workbook_class(self) -> type[Workbook]:
        if False:
            for i in range(10):
                print('nop')
        from pyxlsb import Workbook
        return Workbook

    def load_workbook(self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs) -> Workbook:
        if False:
            while True:
                i = 10
        from pyxlsb import open_workbook
        return open_workbook(filepath_or_buffer, **engine_kwargs)

    @property
    def sheet_names(self) -> list[str]:
        if False:
            print('Hello World!')
        return self.book.sheets

    def get_sheet_by_name(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        self.raise_if_bad_sheet_by_name(name)
        return self.book.get_sheet(name)

    def get_sheet_by_index(self, index: int):
        if False:
            i = 10
            return i + 15
        self.raise_if_bad_sheet_by_index(index)
        return self.book.get_sheet(index + 1)

    def _convert_cell(self, cell) -> Scalar:
        if False:
            print('Hello World!')
        if cell.v is None:
            return ''
        if isinstance(cell.v, float):
            val = int(cell.v)
            if val == cell.v:
                return val
            else:
                return float(cell.v)
        return cell.v

    def get_sheet_data(self, sheet, file_rows_needed: int | None=None) -> list[list[Scalar]]:
        if False:
            return 10
        data: list[list[Scalar]] = []
        previous_row_number = -1
        for row in sheet.rows(sparse=True):
            row_number = row[0].r
            converted_row = [self._convert_cell(cell) for cell in row]
            while converted_row and converted_row[-1] == '':
                converted_row.pop()
            if converted_row:
                data.extend([[]] * (row_number - previous_row_number - 1))
                data.append(converted_row)
                previous_row_number = row_number
            if file_rows_needed is not None and len(data) >= file_rows_needed:
                break
        if data:
            max_width = max((len(data_row) for data_row in data))
            if min((len(data_row) for data_row in data)) < max_width:
                empty_cell: list[Scalar] = ['']
                data = [data_row + (max_width - len(data_row)) * empty_cell for data_row in data]
        return data