"""Module houses `ExcelDispatcher` class, that is used for reading excel files."""
import os
import re
import warnings
from io import BytesIO
import pandas
from modin.config import NPartitions
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher
from modin.pandas.io import ExcelFile
EXCEL_READ_BLOCK_SIZE = 4096

class ExcelDispatcher(TextFileDispatcher):
    """Class handles utils for reading excel files."""

    @classmethod
    def _read(cls, io, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read data from `io` according to the passed `read_excel` `kwargs` parameters.\n\n        Parameters\n        ----------\n        io : str, bytes, ExcelFile, xlrd.Book, path object, or file-like object\n            `io` parameter of `read_excel` function.\n        **kwargs : dict\n            Parameters of `read_excel` function.\n\n        Returns\n        -------\n        new_query_compiler : BaseQueryCompiler\n            Query compiler with imported data for further processing.\n        '
        if kwargs.get('engine', None) is not None and kwargs.get('engine') != 'openpyxl':
            return cls.single_worker_read(io, reason='Modin only implements parallel `read_excel` with `openpyxl` engine, ' + 'please specify `engine=None` or `engine="openpyxl"` to ' + "use Modin's parallel implementation.", **kwargs)
        if kwargs.get('skiprows') is not None:
            return cls.single_worker_read(io, reason="Modin doesn't support 'skiprows' parameter of `read_excel`", **kwargs)
        if isinstance(io, bytes):
            io = BytesIO(io)
        if not isinstance(io, (str, os.PathLike, BytesIO)) or isinstance(io, (ExcelFile, pandas.ExcelFile)):
            if isinstance(io, ExcelFile):
                io._set_pandas_mode()
            return cls.single_worker_read(io, reason='Modin only implements parallel `read_excel` the following types of `io`: ' + 'str, os.PathLike, io.BytesIO.', **kwargs)
        from zipfile import ZipFile
        from openpyxl.reader.excel import ExcelReader
        from openpyxl.worksheet._reader import WorksheetReader
        from openpyxl.worksheet.worksheet import Worksheet
        from modin.core.storage_formats.pandas.parsers import PandasExcelParser
        sheet_name = kwargs.get('sheet_name', 0)
        if sheet_name is None or isinstance(sheet_name, list):
            return cls.single_worker_read(io, reason='`read_excel` functionality is only implemented for a single sheet at a ' + 'time. Multiple sheet reading coming soon!', **kwargs)
        warnings.warn('Parallel `read_excel` is a new feature! If you run into any ' + 'problems, please visit https://github.com/modin-project/modin/issues. ' + "If you find a new issue and can't file it on GitHub, please " + 'email bug_reports@modin.org.')
        io_file = open(io, 'rb') if isinstance(io, (str, os.PathLike)) else io
        try:
            ex = ExcelReader(io_file, read_only=True)
            ex.read()
            wb = ex.wb
            ex.read_manifest()
            ex.read_strings()
            ws = Worksheet(wb)
        finally:
            if isinstance(io, (str, os.PathLike)):
                io_file.close()
        pandas_kw = dict(kwargs)
        with ZipFile(io) as z:
            if isinstance(sheet_name, int):
                sheet_name = 'sheet{}'.format(sheet_name + 1)
            else:
                sheet_name = 'sheet{}'.format(wb.sheetnames.index(sheet_name) + 1)
            if any((sheet_name.lower() in name for name in z.namelist())):
                sheet_name = sheet_name.lower()
            elif any((sheet_name.title() in name for name in z.namelist())):
                sheet_name = sheet_name.title()
            else:
                raise ValueError('Sheet {} not found'.format(sheet_name.lower()))
            kwargs['sheet_name'] = sheet_name
            f = z.open('xl/worksheets/{}.xml'.format(sheet_name))
            f = BytesIO(f.read())
            total_bytes = cls.file_size(f)
            sheet_block = f.read(EXCEL_READ_BLOCK_SIZE)
            end_of_row_tag = b'</row>'
            while end_of_row_tag not in sheet_block:
                sheet_block += f.read(EXCEL_READ_BLOCK_SIZE)
            idx_of_header_end = sheet_block.index(end_of_row_tag) + len(end_of_row_tag)
            sheet_header_with_first_row = sheet_block[:idx_of_header_end]
            if kwargs['header'] is not None:
                f.seek(idx_of_header_end)
                sheet_header = sheet_header_with_first_row
            else:
                start_of_row_tag = b'<row'
                idx_of_header_start = sheet_block.index(start_of_row_tag)
                sheet_header = sheet_block[:idx_of_header_start]
                f.seek(idx_of_header_start)
            kwargs['_header'] = sheet_header
            footer = b'</sheetData></worksheet>'
            common_args = (ws, BytesIO(sheet_header_with_first_row + footer), ex.shared_strings, False)
            if cls.need_rich_text_param():
                reader = WorksheetReader(*common_args, rich_text=False)
            else:
                reader = WorksheetReader(*common_args)
            reader.bind_cells()
            data = PandasExcelParser.get_sheet_data(ws, kwargs.get('convert_float', True))
            if kwargs['header'] is None:
                column_names = pandas.RangeIndex(len(data[0]))
            else:
                column_names = pandas.Index(data[0])
            index_col = kwargs.get('index_col', None)
            if index_col is not None:
                column_names = column_names.drop(column_names[index_col])
            if not all(column_names) or kwargs.get('usecols'):
                pandas_kw['nrows'] = 1
                df = pandas.read_excel(io, **pandas_kw)
                column_names = df.columns
            chunk_size = max(1, (total_bytes - f.tell()) // NPartitions.get())
            (column_widths, num_splits) = cls._define_metadata(pandas.DataFrame(columns=column_names), column_names)
            kwargs['fname'] = io
            kwargs['skiprows'] = 0
            row_count = 0
            data_ids = []
            index_ids = []
            dtypes_ids = []
            kwargs['num_splits'] = num_splits
            while f.tell() < total_bytes:
                args = kwargs
                args['skiprows'] = row_count + args['skiprows']
                args['start'] = f.tell()
                chunk = f.read(chunk_size)
                if b'<row' not in chunk:
                    break
                row_close_tag = b'</row>'
                row_count = re.subn(row_close_tag, b'', chunk)[1]
                while row_count == 0:
                    chunk += f.read(chunk_size)
                    row_count += re.subn(row_close_tag, b'', chunk)[1]
                last_index = chunk.rindex(row_close_tag)
                f.seek(-(len(chunk) - last_index) + len(row_close_tag), 1)
                args['end'] = f.tell()
                if b'</row>' not in chunk and b'</sheetData>' in chunk:
                    break
                remote_results_list = cls.deploy(func=cls.parse, f_kwargs=args, num_returns=num_splits + 2)
                data_ids.append(remote_results_list[:-2])
                index_ids.append(remote_results_list[-2])
                dtypes_ids.append(remote_results_list[-1])
                if b'</sheetData>' in chunk:
                    break
        if index_col is None:
            row_lengths = cls.materialize(index_ids)
            new_index = pandas.RangeIndex(sum(row_lengths))
        else:
            index_objs = cls.materialize(index_ids)
            row_lengths = [len(o) for o in index_objs]
            new_index = index_objs[0].append(index_objs[1:])
        data_ids = cls.build_partition(data_ids, row_lengths, column_widths)
        dtypes = cls.get_dtypes(dtypes_ids, column_names)
        new_frame = cls.frame_cls(data_ids, new_index, column_names, row_lengths, column_widths, dtypes=dtypes)
        new_query_compiler = cls.query_compiler_cls(new_frame)
        if index_col is None:
            new_query_compiler._modin_frame.synchronize_labels(axis=0)
        return new_query_compiler