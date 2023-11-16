import os
import subprocess
import sys
import json

def directory_generator(req, base='/opt/fw/'):
    if False:
        i = 10
        return i + 15
    for versions in req:
        if '/' in versions:
            (pkg, ver) = versions.split('/')
            path = base + pkg + '/' + ver
            if not os.path.exists(path):
                install_pkg(path, pkg + '==' + ver)
        else:
            install_pkg(base + versions, versions)

def install_pkg(path, pkg, base='fw/'):
    if False:
        print('Hello World!')
    if pkg.split('==')[0] if '==' in pkg else pkg == 'torch':
        subprocess.run(f'pip3 install --upgrade {pkg} --target {path} --default-timeout=100 --extra-index-url https://download.pytorch.org/whl/cpu  --no-cache-dir', shell=True)
    elif pkg.split('==')[0] == 'jax':
        subprocess.run(f'pip install --upgrade {pkg} --target  {path}  -f https://storage.googleapis.com/jax-releases/jax_releases.html   --no-cache-dir', shell=True)
    else:
        subprocess.run(f'pip3 install --upgrade {pkg} --target {path} --default-timeout=100   --no-cache-dir', shell=True)

def install_deps(pkgs, path_to_json, base='/opt/fw/'):
    if False:
        while True:
            i = 10
    for fw in pkgs:
        (fw, ver) = fw.split('/')
        path = base + fw + '/' + ver
        with open(path_to_json, 'r') as file:
            json_data = json.load(file)
            for keys in json_data[fw]:
                if isinstance(keys, dict):
                    dep = list(keys.keys())[0]
                    if ver in keys[dep].keys():
                        subprocess.run(f'pip3 install --upgrade {dep}=={keys[dep][ver]} --target {path} --default-timeout=100 --upgrade  --no-cache-dir', shell=True)
                    else:
                        subprocess.run(f'pip3 install  {dep} --target {path} --default-timeout=100   --no-cache-dir', shell=True)
                else:
                    subprocess.run(f"pip3 install  {keys} {(f'-f https://data.pyg.org/whl/torch-{ver}%2Bcpu.html' if keys == 'torch-scatter' else '')} --target {path} --default-timeout=100   --no-cache-dir", shell=True)
if __name__ == '__main__':
    arg_lis = sys.argv
    json_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), 'requirement_mappings_multiversion.json')
    directory_generator(arg_lis[1:])
    install_deps(arg_lis[1:], json_path)