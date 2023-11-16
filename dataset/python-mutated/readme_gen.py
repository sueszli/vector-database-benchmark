"""Generates READMEs using configuration defined in yaml."""
import argparse
import io
import os
import subprocess
import jinja2
import yaml
jinja_env = jinja2.Environment(trim_blocks=True, loader=jinja2.FileSystemLoader(os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))))
README_TMPL = jinja_env.get_template('README.tmpl.rst')

def get_help(file):
    if False:
        print('Hello World!')
    return subprocess.check_output(['python', file, '--help']).decode()

def main():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--destination', default='README.rst')
    args = parser.parse_args()
    source = os.path.abspath(args.source)
    root = os.path.dirname(source)
    destination = os.path.join(root, args.destination)
    jinja_env.globals['get_help'] = get_help
    with io.open(source, 'r') as f:
        config = yaml.safe_load(f)
    os.chdir(root)
    output = README_TMPL.render(config)
    with io.open(destination, 'w') as f:
        f.write(output)
if __name__ == '__main__':
    main()