import inspect
from collections import OrderedDict
from coalib.misc.Enum import enum

class DocstringMetadata:
    _ParseMode = enum('DESCRIPTION', 'PARAM', 'RETVAL')

    def __init__(self, desc, param_dict, retval_desc):
        if False:
            for i in range(10):
                print('nop')
        '\n        Represents a docstring of a python class or function.\n\n        :param desc:        A description as string.\n        :param param_dict:  A dictionary containing parameter names as key and\n                            their description as value. To preserve the order,\n                            use OrderedDict.\n        :param retval_desc: A string describing the return value.\n        '
        self.desc = desc
        self.param_dict = param_dict
        self.retval_desc = retval_desc

    @classmethod
    def from_docstring(cls, docstring):
        if False:
            return 10
        '\n        Parses a python docstring. Usable attributes are:\n        :param\n        @param\n        :return\n        @return\n        '
        lines = inspect.cleandoc(docstring).split('\n')
        parse_mode = cls._ParseMode.DESCRIPTION
        cur_param = ''
        desc = ''
        param_dict = OrderedDict()
        retval_desc = ''
        for line in lines:
            line = line.strip()
            if line.startswith(':param ') or line.startswith('@param '):
                parse_mode = cls._ParseMode.PARAM
                splitted = line[7:].split(':', 1)
                cur_param = splitted[0]
                param_dict[cur_param] = splitted[1].strip()
                continue
            if line.startswith(':return: ') or line.startswith('@return: '):
                parse_mode = cls._ParseMode.RETVAL
                retval_desc = line[9:].strip()
                continue

            def concat_doc_parts(old: str, new: str):
                if False:
                    for i in range(10):
                        print('nop')
                if new != '' and (not old.endswith('\n')):
                    return (old + ' ' + new).strip()
                return old + (new if new != '' else '\n')
            if parse_mode == cls._ParseMode.RETVAL:
                retval_desc = concat_doc_parts(retval_desc, line)
            elif parse_mode == cls._ParseMode.PARAM:
                param_dict[cur_param] = concat_doc_parts(param_dict[cur_param], line)
            else:
                desc = concat_doc_parts(desc, line)
        return cls(desc=desc.strip(), param_dict=param_dict, retval_desc=retval_desc.strip())

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self.desc)