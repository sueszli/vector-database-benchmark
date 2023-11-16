import datetime
from flask_appbuilder._compat import as_unicode

def dict_to_json(xcol, ycols, labels, value_columns):
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts a list of dicts from datamodel query results\n    to google chart json data.\n\n    :param xcol:\n        The name of a string column to be used has X axis on chart\n    :param ycols:\n        A list with the names of series cols, that can be used as numeric\n    :param labels:\n        A dict with the columns labels.\n    :param value_columns:\n        A list of dicts with the values to convert\n    '
    json_data = dict()
    json_data['cols'] = [{'id': xcol, 'label': as_unicode(labels[xcol]), 'type': 'string'}]
    for ycol in ycols:
        json_data['cols'].append({'id': ycol, 'label': as_unicode(labels[ycol]), 'type': 'number'})
    json_data['rows'] = []
    for value in value_columns:
        row = {'c': []}
        if isinstance(value[xcol], datetime.date):
            row['c'].append({'v': str(value[xcol])})
        else:
            row['c'].append({'v': value[xcol]})
        for ycol in ycols:
            if value[ycol]:
                row['c'].append({'v': value[ycol]})
            else:
                row['c'].append({'v': 0})
        json_data['rows'].append(row)
    return json_data