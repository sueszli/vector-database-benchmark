import argparse
from pathlib import Path
import tomlkit
import requests

def get_arrow_sha():
    if False:
        while True:
            i = 10
    url = 'https://api.github.com/repos/apache/arrow-rs/branches/master'
    response = requests.get(url)
    return response.json()['commit']['sha']

def update_commit_dependencies(dependencies, new_sha):
    if False:
        return 10
    if dependencies is None:
        return
    for dep_name in dependencies:
        dep = dependencies[dep_name]
        if hasattr(dep, 'get'):
            if dep.get('git') == 'https://github.com/apache/arrow-rs':
                dep['rev'] = new_sha

def update_commit_cargo_toml(cargo_toml, new_sha):
    if False:
        print('Hello World!')
    print('updating {}'.format(cargo_toml.absolute()))
    with open(cargo_toml) as f:
        data = f.read()
    doc = tomlkit.parse(data)
    update_commit_dependencies(doc.get('dependencies'), new_sha)
    update_commit_dependencies(doc.get('dev-dependencies'), new_sha)
    with open(cargo_toml, 'w') as f:
        f.write(tomlkit.dumps(doc))

def update_version_cargo_toml(cargo_toml, new_version):
    if False:
        i = 10
        return i + 15
    print('updating {}'.format(cargo_toml.absolute()))
    with open(cargo_toml) as f:
        data = f.read()
    doc = tomlkit.parse(data)
    for section in ('dependencies', 'dev-dependencies'):
        for (dep_name, constraint) in doc.get(section, {}).items():
            if dep_name in ('arrow', 'parquet', 'arrow-flight'):
                if type(constraint) == tomlkit.items.String:
                    doc[section][dep_name] = new_version
                elif type(constraint) == dict:
                    doc[section][dep_name]['version'] = new_version
                elif type(constraint) == tomlkit.items.InlineTable:
                    doc[section][dep_name]['version'] = new_version
                else:
                    print('Unknown type for {} {}: {}', dep_name, constraint, type(constraint))
    with open(cargo_toml, 'w') as f:
        f.write(tomlkit.dumps(doc))

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='Update arrow dep versions.')
    sub_parsers = parser.add_subparsers(help='sub-command help')
    parser_version = sub_parsers.add_parser('version', help='update arrow version')
    parser_version.add_argument('new_version', type=str, help='new arrow version')
    parser_version.set_defaults(which='version')
    parser_commit = sub_parsers.add_parser('commit', help='update arrow commit')
    parser_commit.set_defaults(which='commit')
    args = parser.parse_args()
    repo_root = Path(__file__).parent.parent.absolute()
    if args.which == 'version':
        for cargo_toml in repo_root.rglob('Cargo.toml'):
            update_version_cargo_toml(cargo_toml, args.new_version)
    elif args.which == 'commit':
        new_sha = get_arrow_sha()
        print('Updating files in {} to use sha {}'.format(repo_root, new_sha))
        for cargo_toml in repo_root.rglob('Cargo.toml'):
            update_commit_cargo_toml(cargo_toml, new_sha)
if __name__ == '__main__':
    main()