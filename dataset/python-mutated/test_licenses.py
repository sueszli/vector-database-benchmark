import glob
import json
import os
from django.conf import settings
try:
    from pip._internal.req import parse_requirements
except ImportError:
    from pip.req import parse_requirements
from pip._internal.req.constructors import parse_req_from_line

def test_python_and_js_licenses():
    if False:
        print('Hello World!')

    def index_licenses(path):
        if False:
            while True:
                i = 10

        def check_license(license_file):
            if False:
                print('Hello World!')
            with open(license_file) as f:
                data = f.read()
                is_lgpl = 'GNU LESSER GENERAL PUBLIC LICENSE' in data.upper()
                is_gpl = 'GNU GENERAL PUBLIC LICENSE' in data and (not is_lgpl)
                return (is_gpl, is_lgpl)

        def find_embedded_source_version(path, name):
            if False:
                print('Hello World!')
            files = os.listdir(path)
            tgz_files = [f for f in files if f.endswith('.tar.gz')]
            for tgz in tgz_files:
                pkg_name = tgz.split('-')[0].split('_')[0]
                if pkg_name == name:
                    return tgz.split('-')[1].split('.tar.gz')[0]
            return None
        list = {}
        for txt_file in glob.glob('%s/*.txt' % path):
            filename = txt_file.split('/')[-1]
            name = filename[:-4].lower()
            (is_gpl, is_lgpl) = check_license(txt_file)
            list[name] = {'name': name, 'filename': filename, 'gpl': is_gpl, 'source_required': is_gpl or is_lgpl, 'source_version': find_embedded_source_version(path, name)}
        return list

    def read_api_requirements(path):
        if False:
            while True:
                i = 10
        ret = {}
        skip_pbr_license_check = False
        for req_file in ['requirements.txt', 'requirements_git.txt']:
            fname = '%s/%s' % (path, req_file)
            for reqt in parse_requirements(fname, session=''):
                parsed_requirement = parse_req_from_line(reqt.requirement, None)
                name = parsed_requirement.requirement.name
                version = str(parsed_requirement.requirement.specifier)
                if version.startswith('=='):
                    version = version[2:]
                if parsed_requirement.link:
                    if str(parsed_requirement.link).startswith(('http://', 'https://')):
                        (name, version) = str(parsed_requirement.requirement).split('==', 1)
                    else:
                        (name, version) = parsed_requirement.link.filename.split('@', 1)
                    if name.endswith('.git'):
                        name = name[:-4]
                    if name == 'receptor':
                        name = 'receptorctl'
                    if name == 'ansible-runner':
                        skip_pbr_license_check = True
                ret[name] = {'name': name, 'version': version}
        if 'pbr' in ret and skip_pbr_license_check:
            del ret['pbr']
        return ret

    def read_ui_requirements(path):
        if False:
            print('Hello World!')

        def json_deps(jsondata):
            if False:
                for i in range(10):
                    print('nop')
            ret = {}
            deps = jsondata.get('dependencies', {})
            for key in deps.keys():
                key = key.lower()
                devonly = deps[key].get('dev', False)
                if not devonly:
                    if key not in ret.keys():
                        depname = key.replace('/', '-')
                        if depname[0] == '@':
                            depname = depname[1:]
                        ret[depname] = {'name': depname, 'version': deps[key]['version']}
                        ret.update(json_deps(deps[key]))
            return ret
        with open('%s/package-lock.json' % path) as f:
            jsondata = json.load(f)
            return json_deps(jsondata)

    def remediate_licenses_and_requirements(licenses, requirements):
        if False:
            i = 10
            return i + 15
        errors = []
        items = list(licenses.keys())
        items.sort()
        for item in items:
            if item not in [r.lower() for r in requirements.keys()] and item != 'awx':
                errors.append(' license file %s does not correspond to an existing requirement; it should be removed.' % (licenses[item]['filename'],))
                continue
            if licenses[item]['gpl'] and item != 'uwsgi':
                errors.append(' license for %s is GPL. This software cannot be used.' % (item,))
            if licenses[item]['source_required']:
                version = requirements[item]['version']
                if version != licenses[item]['source_version']:
                    errors.append(' embedded source for %s is %s instead of the required version %s' % (item, licenses[item]['source_version'], version))
            elif licenses[item]['source_version']:
                errors.append(' embedded source version %s for %s is included despite not being needed' % (licenses[item]['source_version'], item))
        items = list(requirements.keys())
        items.sort()
        for item in items:
            if item.lower() not in licenses.keys():
                errors.append(' license for requirement %s is missing' % (item,))
        return errors
    base_dir = settings.BASE_DIR
    api_licenses = index_licenses('%s/../licenses' % base_dir)
    ui_licenses = index_licenses('%s/../licenses/ui' % base_dir)
    api_requirements = read_api_requirements('%s/../requirements' % base_dir)
    ui_requirements = read_ui_requirements('%s/ui' % base_dir)
    errors = []
    errors += remediate_licenses_and_requirements(ui_licenses, ui_requirements)
    errors += remediate_licenses_and_requirements(api_licenses, api_requirements)
    if errors:
        raise Exception('Included licenses not consistent with requirements:\n%s' % '\n'.join(errors))