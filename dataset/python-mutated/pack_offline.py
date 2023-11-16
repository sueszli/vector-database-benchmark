urls = [('https://cg.cs.tsinghua.edu.cn/jittor/assets/dnnl_lnx_2.2.0_cpu_gomp.tgz', 'dnnl_lnx_2.2.0_cpu_gomp.tgz'), ('https://cg.cs.tsinghua.edu.cn/jittor/assets/dnnl_lnx_2.2.0_cpu_gomp_aarch64.tgz', 'dnnl_lnx_2.2.0_cpu_gomp_aarch64.tgz'), ('https://codeload.github.com/NVIDIA/cub/tar.gz/1.11.0', 'cub-1.11.0.tgz'), ('https://codeload.github.com/Jittor/cutt/zip/v1.2', 'cutt-1.2.zip'), ('https://codeload.github.com/NVIDIA/nccl/tar.gz/v2.8.4-1', 'nccl.tgz'), ('https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz', 'train-images-idx3-ubyte.gz'), ('https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz', 'train-labels-idx1-ubyte.gz'), ('https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz'), ('https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz')]
import urllib
from pathlib import Path
import os
import glob
import shutil
import sys
cpath = os.path.join(str(Path.home()), '.cache', 'jittor', 'offpack')
os.makedirs(cpath + '/python/jittor_offline', exist_ok=True)
for (url, file_path) in urls:
    file_path = os.path.join(cpath, 'python/jittor_offline', file_path)
    print('download', url, file_path)
    urllib.request.urlretrieve(url, file_path)
with open(os.path.join(cpath, 'MANIFEST.in'), 'w') as f:
    f.write('include python/jittor_offline/*')
with open(os.path.join(cpath, '__init__.py'), 'w') as f:
    f.write('')
with open(os.path.join(cpath, 'setup.py'), 'w') as f:
    f.write('\nimport setuptools\n\n\nsetuptools.setup(\n    name="jittor_offline",\n    version="0.0.7",\n    author="jittor",\n    author_email="jittor@qq.com",\n    description="jittor project",\n    long_description="jittor_offline",\n    long_description_content_type="text/markdown",\n    url="https://github.com/jittor/jittor",\n    project_urls={\n        "Bug Tracker": "https://github.com/jittor/jittor/issues",\n    },\n    classifiers=[\n        "Programming Language :: Python :: 3",\n        "Operating System :: OS Independent",\n    ],\n    packages=["jittor_offline"],\n    package_dir={"": "python"},\n    package_data={\'\': [\'*\', \'*/*\', \'*/*/*\',\'*/*/*/*\',\'*/*/*/*/*\',\'*/*/*/*/*/*\']},\n    python_requires=">=3.7",\n    install_requires=[\n        "jittor>=1.3.4.16",\n    ],\n)\n')

def callback(func, path, exc_info):
    if False:
        for i in range(10):
            print('nop')
    print(f'remove "{path}" failed.')

def rmtree(path):
    if False:
        i = 10
        return i + 15
    if os.path.isdir(path):
        print(f'remove "{path}" recursive.')
        shutil.rmtree(path, onerror=callback)

def remove_tmpfile():
    if False:
        print('Hello World!')
    dist_file = home_path + '/dist'
    egg_file = glob.glob(home_path + '/**/*egg-info')
    rmtree(dist_file)
    for e in egg_file:
        rmtree(e)

def run_cmd(cmd):
    if False:
        while True:
            i = 10
    print('[CMD]', cmd)
    assert os.system(cmd) == 0
home_path = cpath
os.chdir(cpath)
remove_tmpfile()
run_cmd(f'{sys.executable} ./setup.py sdist')
run_cmd(f'{sys.executable} -m twine upload dist/*')
remove_tmpfile()