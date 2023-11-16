MAX_NUM_FIELDS = 20
HEADER = 'id: variable_struct\nlabel: Struct Variable\nflags: [ show_id ]\n\nparameters:\n'
TEMPLATES = '\ntemplates:\n    imports: "def struct(data): return type(\'Struct\', (object,), data)()"\n    var_make: |-\n        self.${{id}} = ${{id}} = struct({{\n            % for i in range({0}):\n            <%\n                field = context.get(\'field\' + str(i))\n                value = context.get(\'value\' + str(i))\n            %>\n            % if len(str(field)) > 2:\n            ${{field}}: ${{value}},\n            % endif\n            % endfor\n        }})\n    var_value: |-\n        struct({{\n            % for i in range({0}):\n            <%\n                field = context.get(\'field\' + str(i))\n            %>\n            % if len(str(field)) > 2:\n            ${{field}}: ${{field}},\n            % endif\n            % endfor\n        }})\n'
FIELD0 = '-   id: field0\n    label: Field 0\n    category: Fields\n    dtype: string\n    default: field0\n    hide: part\n'
FIELDS = '-   id: field{0}\n    label: Field {0}\n    category: Fields\n    dtype: string\n    hide: part\n'
VALUES = "-   id: value{0}\n    label: ${{field{0}}}\n    dtype: raw\n    default: '0'\n    hide: ${{ 'none' if field{0} else 'all' }}\n"
ASSERTS = '- ${{ (str(field{0}) or "a")[0].isalpha() }}\n- ${{ (str(field{0}) or "a").isalnum() }}\n'
FOOTER = '\ndocumentation: |-\n    This is a simple struct/record like variable.\n\n    Attribute/field names can be specified in the tab \'Fields\'.\n    For each non-empty field a parameter with type raw is shown.\n    Value access via the dot operator, e.g. "variable_struct_0.field0"\n\nfile_format: 1\n'

def make_yml(num_fields):
    if False:
        print('Hello World!')
    return ''.join((HEADER.format(num_fields), FIELD0, ''.join((FIELDS.format(i) for i in range(1, num_fields))), ''.join((VALUES.format(i) for i in range(num_fields))), 'value: ${value}\n\nasserts:\n', ''.join((ASSERTS.format(i) for i in range(num_fields))), ''.join(TEMPLATES.format(num_fields)), FOOTER))
if __name__ == '__main__':
    import sys
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = __file__[:-3]
    data = make_yml(MAX_NUM_FIELDS)
    with open(filename, 'wb') as fp:
        fp.write(data.encode())