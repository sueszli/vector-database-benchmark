import fileinput
import os
import shutil
import sys
import requests
from pylint.pyreverse.main import Run
here = os.path.dirname(__file__)
plantuml_jarfile_url = 'https://sourceforge.net/projects/plantuml/files/plantuml.jar/download'
ignore = ['_vendor', 'compat.py', 'misc.py', 'utils.py', 'exports.py', 'api.py']
extra_puml = '\n'.join(['left to right direction', 'skinparam nodesep 5', 'skinparam ranksep 5'])
replacements = (('set namespaceSeparator none', f'set namespaceSeparator none\n{extra_puml}'), ('class "" as conda.common.compat.six_with_metaclass.', 'class "six_with_metaclass" as conda.common.compat.six_with_metaclass.'))

def post_process(files, output_path):
    if False:
        while True:
            i = 10
    'Replace all items from the replacements list above in the given files.'
    for file in files:
        with fileinput.input(files=[os.path.join(output_path, file)], inplace=True) as open_file:
            for line in open_file:
                for (old, new) in replacements:
                    line = line.replace(old, new)
                sys.stdout.write(line)

def generate_pumls(app=None, config=None):
    if False:
        return 10
    '\n    Generates PlantUML files for the given packages and writes\n    the files to the components directory in the documentation source.\n    '
    sys.stdout.write('Generating PlantUML...\n')
    sys.stdout.flush()
    packages = ['conda']
    for package in packages:
        output_path = os.path.join(here, '..', 'dev-guide', 'umls')
        output_format = 'puml'
        files = [f'packages_{package}.{output_format}', f'classes_{package}.{output_format}']
        ignore_list = ','.join(ignore)
        args = [package, f'--ignore={ignore_list}', f'--output={output_format}', '--colorized', '--max-color-depth=8', f'--project={package}', f'--output-directory={output_path}', '--all-associated', '--all-ancestors']
        try:
            Run(args)
        except SystemExit as err:
            if err.code:
                raise
        post_process(files, output_path)
        sys.stdout.write('Done generating PlantUML files.\n')

def download_plantuml(app, config):
    if False:
        print('Hello World!')
    if os.path.exists(config.plantuml_jarfile_path):
        sys.stdout.write(f'PlantUML jar file already downloaded. To update run `make clean` or manually delete {config.plantuml_jarfile_path}.\n')
    else:
        parent = os.path.dirname(config.plantuml_jarfile_path)
        if not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with requests.get(plantuml_jarfile_url, stream=True) as response:
            sys.stdout.write(f'Downloading PlantUML jar file to {config.plantuml_jarfile_path}...')
            sys.stdout.flush()
            response.raise_for_status()
            response.raw.decode_content = True
            with open(config.plantuml_jarfile_path, 'wb') as jarfile:
                shutil.copyfileobj(response.raw, jarfile)
                sys.stdout.write('done.\n')

def setup(app):
    if False:
        return 10
    if 'AUTOBUILD' not in os.environ:
        app.add_config_value('plantuml_jarfile_path', None, rebuild='')
        app.connect('config-inited', download_plantuml)
        app.connect('config-inited', generate_pumls)
    return {'version': '0.1', 'parallel_read_safe': False, 'parallel_write_safe': False}
if __name__ == '__main__':
    generate_pumls()