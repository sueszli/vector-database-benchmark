"""
This script is used to automate generation of requirements.txt for st2 components.

The idea behind this script is that that each component has it's own requirements
"in-requirements.txt" file. in-requirements.txt is an input requirements file -
a requirements file with dependencies but WITHOUT any version restrictions.

In addition to this file, there's also the top-level "fixed-requirements.txt"
which pins production versions for the whole st2 stack. During production use
(building, packaging, etc) requirements.txt is generated from in-requirements.txt
where version of packages are fixed according to fixed-requirements.txt.
"""
from __future__ import absolute_import, print_function
import argparse
import os
import os.path
import sys
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
if PY3:
    text_type = str
else:
    text_type = unicode
OSCWD = os.path.abspath(os.curdir)
GET_PIP = '    curl https://bootstrap.pypa.io/get-pip.py | python'
try:
    from pip import __version__ as pip_version
except ImportError as e:
    print('Failed to import pip: %s' % text_type(e))
    print('')
    print('Download pip:\n%s' % GET_PIP)
    sys.exit(1)
try:
    from pip.req import parse_requirements
except ImportError:
    try:
        from pip._internal.req.req_file import parse_requirements
    except ImportError as e:
        print('Failed to import parse_requirements from pip: %s' % text_type(e))
        print('Using pip: %s' % str(pip_version))
        sys.exit(1)
try:
    from pip._internal.req.constructors import parse_req_from_line
except ImportError:
    pass

def parse_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='Tool for requirements.txt generation.')
    parser.add_argument('-s', '--source-requirements', nargs='+', required=True, help='Specify paths to requirements file(s). In case several requirements files are given their content is merged.')
    parser.add_argument('-f', '--fixed-requirements', required=True, help='Specify path to fixed-requirements.txt file.')
    parser.add_argument('-o', '--output-file', default='requirements.txt', help='Specify path to the resulting requirements file.')
    parser.add_argument('--skip', default=None, help='Comma delimited list of requirements to not include in the generated file.')
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    return vars(parser.parse_args())

def load_requirements(file_path):
    if False:
        for i in range(10):
            print('nop')
    return tuple((r for r in parse_requirements(file_path, session=False)))

def locate_file(path, must_exist=False):
    if False:
        for i in range(10):
            print('nop')
    if not os.path.isabs(path):
        path = os.path.join(OSCWD, path)
    if must_exist and (not os.path.isfile(path)):
        print("Error: couldn't locate file `{0}'".format(path))
    return path

def merge_source_requirements(sources):
    if False:
        while True:
            i = 10
    "\n    Read requirements source files and merge it's content.\n    "
    projects = set()
    merged_requirements = []
    for infile_path in (locate_file(p, must_exist=True) for p in sources):
        for req in load_requirements(infile_path):
            if hasattr(req, 'requirement'):
                parsedreq = parse_req_from_line(req.requirement, req.line_source)
                if parsedreq.requirement:
                    if parsedreq.requirement.name in projects:
                        continue
                    projects.add(parsedreq.requirement.name)
                    merged_requirements.append(req)
                elif parsedreq.link:
                    merged_requirements.append(req)
                else:
                    raise RuntimeError('Unexpected requirement {0}'.format(req))
            elif req.req:
                if req.name in projects:
                    continue
                projects.add(req.name)
                merged_requirements.append(req)
            elif req.link:
                merged_requirements.append(req)
            else:
                raise RuntimeError('Unexpected requirement {0}'.format(req))
    return merged_requirements

def write_requirements(sources=None, fixed_requirements=None, output_file=None, skip=None):
    if False:
        return 10
    '\n    Write resulting requirements taking versions from the fixed_requirements.\n    '
    skip = skip or []
    requirements = merge_source_requirements(sources)
    fixed = load_requirements(locate_file(fixed_requirements, must_exist=True))
    fixedreq_hash = {}
    for req in fixed:
        if hasattr(req, 'requirement'):
            parsedreq = parse_req_from_line(req.requirement, req.line_source)
            project_name = parsedreq.requirement.name
            if not req.requirement:
                continue
        else:
            project_name = req.name
            if not req.req:
                continue
        if project_name in fixedreq_hash:
            raise ValueError('Duplicate definition for dependency "%s"' % project_name)
        fixedreq_hash[project_name] = req
    lines_to_write = []
    links = set()
    for req in requirements:
        if hasattr(req, 'requirement'):
            parsedreq = parse_req_from_line(req.requirement, req.line_source)
            project_name = parsedreq.requirement.name
            linkreq = parsedreq
        else:
            project_name = req.name
            linkreq = req
        if project_name in skip:
            continue
        if linkreq.link and linkreq.link not in links:
            links.add(linkreq.link)
            if hasattr(req, 'req') and req.req and (str(req.req).count('@') == 2):
                rline = str(linkreq.req)
                if linkreq.markers:
                    rline += ' ; {}'.format(str(linkreq.markers))
            else:
                rline = str(linkreq.link)
            if hasattr(req, 'is_editable') and req.is_editable or (hasattr(req, 'editable') and req.editable):
                rline = '-e %s' % rline
        elif hasattr(req, 'requirement') and req.requirement:
            project = parsedreq.requirement.name
            req_obj = fixedreq_hash.get(project, req)
            rline = str(req_obj.requirement)
        elif hasattr(req, 'req') and req.req:
            project = req.name
            req_obj = fixedreq_hash.get(project, req)
            rline = str(req_obj.req)
            if req_obj.markers:
                rline += ' ; {}'.format(str(req_obj.markers))
        lines_to_write.append(rline)
    lines_to_write = sorted(lines_to_write)
    data = '\n'.join(lines_to_write) + '\n'
    with open(output_file, 'w') as fp:
        fp.write("# Don't edit this file. It's generated automatically!\n")
        fp.write('# If you want to update global dependencies, modify fixed-requirements.txt\n')
        fp.write("# and then run 'make requirements' to update requirements.txt for all\n")
        fp.write('# components.\n')
        fp.write('# If you want to update depdencies for a single component, modify the\n')
        fp.write("# in-requirements.txt for that component and then run 'make requirements' to\n")
        fp.write('# update the component requirements.txt\n')
        fp.write(data)
    print('Requirements written to: {0}'.format(output_file))
if __name__ == '__main__':
    args = parse_args()
    if args['skip']:
        skip = args['skip'].split(',')
    else:
        skip = None
    write_requirements(sources=args['source_requirements'], fixed_requirements=args['fixed_requirements'], output_file=args['output_file'], skip=skip)