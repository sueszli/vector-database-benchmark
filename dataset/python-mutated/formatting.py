from typing import List

def compute_columns_width(data: List[List[str]], padding: int=2) -> List[int]:
    if False:
        return 10
    'Compute columns width for display purposes:\n    Find size for each columns in the data and add padding.\n    Args:\n        data (List[List[str]]): Tabular data containing rows and columns.\n        padding (int): Number of character to adds to create space between columns.\n    Returns:\n        columns_width (List[int]): The computed columns widths for each column according to input data.\n    '
    columns_width = [0 for _ in data[0]]
    for row in data:
        for (i, col) in enumerate(row):
            current_col_width = len(col) + padding
            if current_col_width > columns_width[i]:
                columns_width[i] = current_col_width
    return columns_width

def camelcased_to_uppercased_spaced(camelcased: str) -> str:
    if False:
        print('Hello World!')
    'Util function to transform a camelCase string to a UPPERCASED SPACED string\n    e.g: dockerImageName -> DOCKER IMAGE NAME\n    Args:\n        camelcased (str): The camel cased string to convert.\n\n    Returns:\n        (str): The converted UPPERCASED SPACED string\n    '
    return ''.join(map(lambda x: x if x.islower() else ' ' + x, camelcased)).upper()

def display_as_table(data: List[List[str]]) -> str:
    if False:
        print('Hello World!')
    'Formats tabular input data into a displayable table with columns.\n    Args:\n        data (List[List[str]]): Tabular data containing rows and columns.\n    Returns:\n        table (str): String representation of input tabular data.\n    '
    columns_width = compute_columns_width(data)
    table = '\n'.join([''.join((col.ljust(columns_width[i]) for (i, col) in enumerate(row))) for row in data])
    return table

def format_column_names(camelcased_column_names: List[str]) -> List[str]:
    if False:
        print('Hello World!')
    'Format camel cased column names to uppercased spaced column names\n\n    Args:\n        camelcased_column_names (List[str]): Column names in camel case.\n\n    Returns:\n        (List[str]): Column names in uppercase with spaces.\n    '
    return [camelcased_to_uppercased_spaced(column_name) for column_name in camelcased_column_names]