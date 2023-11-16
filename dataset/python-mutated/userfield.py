"""Class to show and manipulate user fields in odf documents."""
import sys
import zipfile
from odf.text import UserFieldDecl
from odf.namespaces import OFFICENS
from odf.opendocument import load
OUTENCODING = 'utf-8'
VALUE_TYPES = {'float': (OFFICENS, 'value'), 'percentage': (OFFICENS, 'value'), 'currency': (OFFICENS, 'value'), 'date': (OFFICENS, 'date-value'), 'time': (OFFICENS, 'time-value'), 'boolean': (OFFICENS, 'boolean-value'), 'string': (OFFICENS, 'string-value')}

class UserFields:
    """List, view and manipulate user fields."""
    src_file = None
    dest_file = None

    def __init__(self, src=None, dest=None):
        if False:
            i = 10
            return i + 15
        'Constructor\n\n        src ... source document name, file like object or None for stdin\n        dest ... destination document name, file like object or None for stdout\n\n        '
        self.src_file = src
        self.dest_file = dest
        self.document = None

    def loaddoc(self):
        if False:
            return 10
        if isinstance(self.src_file, (bytes, str)):
            if not zipfile.is_zipfile(self.src_file):
                raise TypeError('%s is no odt file.' % self.src_file)
        elif self.src_file is None:
            self.src_file = sys.stdin
        self.document = load(self.src_file)

    def savedoc(self):
        if False:
            return 10
        if self.dest_file is None:
            self.document.save('-')
        else:
            self.document.save(self.dest_file)

    def list_fields(self):
        if False:
            return 10
        'List (extract) all known user-fields.\n\n        Returns list of user-field names.\n\n        '
        return [x[0] for x in self.list_fields_and_values()]

    def list_fields_and_values(self, field_names=None):
        if False:
            print('Hello World!')
        'List (extract) user-fields with type and value.\n\n        field_names ... list of field names to show or None for all.\n\n        Returns list of tuples (<field name>, <field type>, <value>).\n\n        '
        self.loaddoc()
        found_fields = []
        all_fields = self.document.getElementsByType(UserFieldDecl)
        for f in all_fields:
            value_type = f.getAttribute('valuetype')
            if value_type == 'string':
                value = f.getAttribute('stringvalue')
            else:
                value = f.getAttribute('value')
            field_name = f.getAttribute('name')
            if field_names is None or field_name in field_names:
                found_fields.append((field_name.encode(OUTENCODING), value_type.encode(OUTENCODING), value.encode(OUTENCODING)))
        return found_fields

    def list_values(self, field_names):
        if False:
            while True:
                i = 10
        'Extract the contents of given field names from the file.\n\n        field_names ... list of field names\n\n        Returns list of field values.\n\n        '
        return [x[2] for x in self.list_fields_and_values(field_names)]

    def get(self, field_name):
        if False:
            i = 10
            return i + 15
        'Extract the contents of this field from the file.\n\n        Returns field value or None if field does not exist.\n\n        '
        values = self.list_values([field_name])
        if not values:
            return None
        return values[0]

    def get_type_and_value(self, field_name):
        if False:
            i = 10
            return i + 15
        'Extract the type and contents of this field from the file.\n\n        Returns tuple (<type>, <field-value>) or None if field does not exist.\n\n        '
        fields = self.list_fields_and_values([field_name])
        if not fields:
            return None
        (field_name, value_type, value) = fields[0]
        return (value_type, value)

    def update(self, data):
        if False:
            while True:
                i = 10
        'Set the value of user fields. The field types will be the same.\n\n        data ... dict, with field name as key, field value as value\n\n        Returns None\n\n        '
        self.loaddoc()
        all_fields = self.document.getElementsByType(UserFieldDecl)
        for f in all_fields:
            field_name = f.getAttribute('name')
            if field_name in data:
                value_type = f.getAttribute('valuetype')
                value = data.get(field_name)
                if value_type == 'string':
                    f.setAttribute('stringvalue', value)
                else:
                    f.setAttribute('value', value)
        self.savedoc()