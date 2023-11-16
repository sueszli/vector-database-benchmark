import os
import subprocess
import sys
subprocess.run(f'pip3 install requests', shell=True)
import requests

def get_latest_package_version(package_name):
    if False:
        print('Hello World!')
    try:
        url = f'https://pypi.org/pypi/{package_name}/json'
        response = requests.get(url)
        response.raise_for_status()
        package_info = response.json()
        return package_info['info']['version']
    except requests.exceptions.RequestException as e:
        print(f'Error: Failed to fetch package information for {package_name}.')
        return None

def directory_generator(req, base='/opt/fw/'):
    if False:
        for i in range(10):
            print('nop')
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
        i = 10
        return i + 15
    if pkg.split('==')[0] if '==' in pkg else pkg == 'torch':
        subprocess.run(f'yes |pip3 install --upgrade {pkg} --target {path} --default-timeout=100 --extra-index-url https://download.pytorch.org/whl/cu118  --no-cache-dir', shell=True)
    elif pkg.split('==')[0] if '==' in pkg else pkg == 'jax':
        subprocess.run(f"yes |pip install --upgrade --target {path} 'jax[cuda11_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html   --no-cache-dir", shell=True)
    elif pkg.split('==')[0] if '==' in pkg else pkg == 'paddle':
        subprocess.run(f"yes |pip install  paddlepaddle-gpu=={get_latest_package_version('paddlepaddle')}.post117 --target {path}  -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html   --no-cache-dir", shell=True)
    elif pkg.split('==')[0] if '==' in pkg else pkg == 'tensorflow':
        subprocess.run(f'yes |pip install tensorflow[and-cuda] --target {path}', shell=True)
    else:
        subprocess.run(f'yes |pip3 install --upgrade {pkg} --target {path} --default-timeout=100   --no-cache-dir', shell=True)
if __name__ == '__main__':
    arg_lis = sys.argv
    if len(arg_lis) > 1:
        directory_generator(arg_lis[1:], '')
    else:
        directory_generator(['tensorflow', 'jax', 'torch', 'paddle'])
    subprocess.run(f'yes |pip3 uninstall requests', shell=True)