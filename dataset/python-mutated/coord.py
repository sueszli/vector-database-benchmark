"""
Generates libopenage/coord/coord_{xy, xyz, ne_se, ne_se_up}.{h, cpp}
"""
from mako.template import Template

def generate_coord_basetypes(projectdir):
    if False:
        i = 10
        return i + 15
    '\n    Generates the test/demo method symbol lookup file from tests_cpp.\n\n    projectdir is a util.fslike.path.Path.\n    '
    member_lists = [['x', 'y'], ['x', 'y', 'z'], ['ne', 'se'], ['ne', 'se', 'up']]
    template_files_spec = [('libopenage/coord/coord.h.template', "libopenage/coord/coord_${''.join(members)}.gen.h"), ('libopenage/coord/coord.cpp.template', "libopenage/coord/coord_${''.join(members)}.gen.cpp")]
    templates = []
    for (template_filename, output_filename) in template_files_spec:
        with projectdir.joinpath(template_filename).open() as template_file:
            templates.append((Template(template_file.read()), Template(output_filename)))
    for member_list in member_lists:

        def format_members(formatstring, join_with=', '):
            if False:
                for i in range(10):
                    print('nop')
            '\n            For being called by the template engine.\n\n            >>> format_members("{0} = {0}")\n            "x = x, y = y"\n            '
            return join_with.join((formatstring.format(m) for m in member_list))
        template_dict = {'members': member_list, 'formatted_members': format_members, 'camelcase': ''.join((member.title() for member in member_list))}
        for (template, output_filename_template) in templates:
            output_filename = output_filename_template.render(**template_dict)
            with projectdir.joinpath(output_filename).open('w') as output_file:
                output = template.render(**template_dict)
                output_file.write(output)
                if not output.endswith('\n'):
                    output_file.write('\n')