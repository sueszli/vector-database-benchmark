"""jc - JSON Convert universal parsers"""
from typing import Iterable, List, Dict

def simple_table_parse(data: Iterable[str]) -> List[Dict]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Parse simple tables. There should be no blank cells. The last column\n    may contain data with spaces.\n\n    Example Table:\n\n        col_1     col_2     col_3     col_4     col_5\n        apple     orange    pear      banana    my favorite fruits\n        carrot    squash    celery    spinach   my favorite veggies\n        chicken   beef      pork      eggs      my favorite proteins\n\n        [{'col_1': 'apple', 'col_2': 'orange', 'col_3': 'pear', 'col_4':\n        'banana', 'col_5': 'my favorite fruits'}, {'col_1': 'carrot',\n        'col_2': 'squash', 'col_3': 'celery', 'col_4': 'spinach', 'col_5':\n        'my favorite veggies'}, {'col_1': 'chicken', 'col_2': 'beef',\n        'col_3': 'pork', 'col_4': 'eggs', 'col_5': 'my favorite proteins'}]\n\n    Parameters:\n\n        data:   (iter)   Text data to parse that has been split into lines\n                         via .splitlines(). Item 0 must be the header row.\n                         Any spaces in header names should be changed to\n                         underscore '_'. You should also ensure headers are\n                         lowercase by using .lower().\n\n                         Also, ensure there are no blank rows in the data.\n\n    Returns:\n\n        List of Dictionaries\n    "
    data = list(data)
    headers = [h for h in ' '.join(data[0].strip().split()).split() if h]
    raw_data = map(lambda s: s.strip().split(None, len(headers) - 1), data[1:])
    raw_output = [dict(zip(headers, r)) for r in raw_data]
    return raw_output

def sparse_table_parse(data: Iterable[str], delim: str='\u2063') -> List[Dict]:
    if False:
        i = 10
        return i + 15
    "\n    Parse tables with missing column data or with spaces in column data.\n    Blank cells are converted to None in the resulting dictionary. Data\n    elements must line up within column boundaries.\n\n    Example Table:\n\n        col_1        col_2     col_3     col_4         col_5\n        apple        orange              fuzzy peach   my favorite fruits\n        green beans            celery    spinach       my favorite veggies\n        chicken      beef                brown eggs    my favorite proteins\n\n        [{'col_1': 'apple', 'col_2': 'orange', 'col_3': None, 'col_4':\n        'fuzzy peach', 'col_5': 'my favorite fruits'}, {'col_1':\n        'green beans', 'col_2': None, 'col_3': 'celery', 'col_4': 'spinach',\n        'col_5': 'my favorite veggies'}, {'col_1': 'chicken', 'col_2':\n        'beef', 'col_3': None, 'col_4': 'brown eggs', 'col_5':\n        'my favorite proteins'}]\n\n    Parameters:\n\n        data:   (iter)   An iterable of string lines (e.g. str.splitlines())\n                         Item 0 must be the header row. Any spaces in header\n                         names should be changed to underscore '_'. You\n                         should also ensure headers are lowercase by using\n                         .lower(). Do not change the position of header\n                         names as the positions are used to find the data.\n\n                         Also, ensure there are no blank line items.\n\n        delim:  (string) Delimiter to use. By default `u\\2063`\n                         (invisible separator) is used since it is unlikely\n                         to ever be seen in terminal output. You can change\n                         this for troubleshooting purposes or if there is a\n                         delimiter conflict with your data.\n\n    Returns:\n\n        List of Dictionaries\n    "
    data = list(data)
    max_len = max([len(x) for x in data])
    new_data = []
    for line in data:
        new_data.append(line + ' ' * (max_len - len(line)))
    data = new_data
    output: List = []
    header_text: str = data.pop(0)
    header_text = header_text + ' '
    header_list: List = header_text.split()
    header_search = [header_list[0]]
    for h in header_list[1:]:
        header_search.append(' ' + h + ' ')
    header_spec_list = []
    for (i, column) in enumerate(header_list[0:len(header_list) - 1]):
        header_spec = {'name': column, 'end': header_text.find(header_search[i + 1])}
        header_spec_list.append(header_spec)
    if data:
        for entry in data:
            output_line = {}
            for col in reversed(header_list):
                for h_spec in header_spec_list:
                    if h_spec['name'] == col:
                        h_end = h_spec['end']
                        while h_end > 0 and (not entry[h_end].isspace()):
                            h_end -= 1
                        entry = entry[:h_end] + delim + entry[h_end + 1:]
            entry_list = entry.split(delim, maxsplit=len(header_list) - 1)
            clean_entry_list = []
            for col in entry_list:
                clean_entry = col.strip()
                if clean_entry == '':
                    clean_entry = None
                clean_entry_list.append(clean_entry)
            output_line = dict(zip(header_list, clean_entry_list))
            output.append(output_line)
    return output