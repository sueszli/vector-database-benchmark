import os
from calibre.ebooks.rtf2xml import copy
from calibre.ptempfile import better_mktemp
from . import open_for_read, open_for_write
'\nSimply write the list of strings after style table\n'

class BodyStyles:
    """
    Insert table data for tables.
    Logic:
    """

    def __init__(self, in_file, list_of_styles, bug_handler, copy=None, run_level=1):
        if False:
            i = 10
            return i + 15
        "\n        Required:\n            'file'--file to parse\n            'table_data' -- a dictionary for each table.\n        Optional:\n            'copy'-- whether to make a copy of result for debugging\n            'temp_dir' --where to output temporary results (default is\n            directory from which the script is run.)\n        Returns:\n            nothing\n            "
        self.__file = in_file
        self.__bug_handler = bug_handler
        self.__copy = copy
        self.__list_of_styles = list_of_styles
        self.__run_level = run_level
        self.__write_to = better_mktemp()

    def insert_info(self):
        if False:
            while True:
                i = 10
        '\n        '
        read_obj = open_for_read(self.__file)
        self.__write_obj = open_for_write(self.__write_to)
        line_to_read = 1
        while line_to_read:
            line_to_read = read_obj.readline()
            line = line_to_read
            if line == 'mi<tg<close_____<style-table\n':
                if len(self.__list_of_styles) > 0:
                    self.__write_obj.write('mi<tg<open______<styles-in-body\n')
                    the_string = ''.join(self.__list_of_styles)
                    self.__write_obj.write(the_string)
                    self.__write_obj.write('mi<tg<close_____<styles-in-body\n')
                elif self.__run_level > 3:
                    msg = 'Not enough data for each table\n'
                    raise self.__bug_handler(msg)
            self.__write_obj.write(line)
        read_obj.close()
        self.__write_obj.close()
        copy_obj = copy.Copy(bug_handler=self.__bug_handler)
        if self.__copy:
            copy_obj.copy_file(self.__write_to, 'body_styles.data')
        copy_obj.rename(self.__write_to, self.__file)
        os.remove(self.__write_to)