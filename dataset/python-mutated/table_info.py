import os
from calibre.ebooks.rtf2xml import copy
from calibre.ptempfile import better_mktemp
from . import open_for_read, open_for_write
'\n'

class TableInfo:
    """
    Insert table data for tables.
    Logic:
    """

    def __init__(self, in_file, bug_handler, table_data, copy=None, run_level=1):
        if False:
            while True:
                i = 10
        "\n        Required:\n            'file'--file to parse\n            'table_data' -- a dictionary for each table.\n        Optional:\n            'copy'-- whether to make a copy of result for debugging\n            'temp_dir' --where to output temporary results (default is\n            directory from which the script is run.)\n        Returns:\n            nothing\n            "
        self.__file = in_file
        self.__bug_handler = bug_handler
        self.__copy = copy
        self.__table_data = table_data
        self.__run_level = run_level
        self.__write_to = better_mktemp()

    def insert_info(self):
        if False:
            return 10
        '\n        '
        read_obj = open_for_read(self.__file)
        self.__write_obj = open_for_write(self.__write_to)
        line_to_read = 1
        while line_to_read:
            line_to_read = read_obj.readline()
            line = line_to_read
            if line == 'mi<mk<tabl-start\n':
                if len(self.__table_data) > 0:
                    table_dict = self.__table_data[0]
                    self.__write_obj.write('mi<tg<open-att__<table')
                    keys = table_dict.keys()
                    for key in keys:
                        self.__write_obj.write(f'<{key}>{table_dict[key]}')
                    self.__write_obj.write('\n')
                    self.__table_data = self.__table_data[1:]
                else:
                    if self.__run_level > 3:
                        msg = 'Not enough data for each table\n'
                        raise self.__bug_handler(msg)
                    self.__write_obj.write('mi<tg<open______<table\n')
            elif line == 'mi<mk<table-end_\n':
                self.__write_obj.write('mi<tg<close_____<table\n')
            self.__write_obj.write(line)
        read_obj.close()
        self.__write_obj.close()
        copy_obj = copy.Copy(bug_handler=self.__bug_handler)
        if self.__copy:
            copy_obj.copy_file(self.__write_to, 'table_info.data')
        copy_obj.rename(self.__write_to, self.__file)
        os.remove(self.__write_to)