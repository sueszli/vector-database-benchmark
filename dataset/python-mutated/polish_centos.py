import jittor as jt
import os
import jittor_utils as jit_utils
home_path = jit_utils.home()

def run_cmd(cmd):
    if False:
        for i in range(10):
            print('nop')
    print('RUN CMD:', cmd)
    assert os.system(cmd) == 0

def run_in_centos(env):
    if False:
        i = 10
        return i + 15
    dockerfile_src = "\n    FROM centos:7\n\n    WORKDIR /root\n\n    # install python\n    RUN yum install gcc openssl-devel bzip2-devel libffi-devel zlib-devel wget -y\n    RUN wget https://www.python.org/ftp/python/3.8.3/Python-3.8.3.tgz\n    RUN tar xzf Python-3.8.3.tgz\n    RUN yum install make -y\n    RUN cd Python-3.8.3 && ./configure --enable-optimizations && make altinstall -j8\n\n    # install g++-7\n    # or yum install gcc-g++\n    RUN yum install centos-release-scl -y\n    RUN yum install devtoolset-7-gcc-c++ -y\n    RUN yum install which -y\n    RUN scl enable devtoolset-7 'g++ --version'\n    RUN python3.8 -m pip install numpy tqdm pillow astunparse\n    "
    with open('/tmp/centos_build_env', 'w') as f:
        f.write(dockerfile_src)
    centos_path = os.path.join(home_path, '.cache', 'centos')
    os.makedirs(centos_path + '/src/jittor', exist_ok=True)
    os.makedirs(centos_path + '/src/jittor_utils', exist_ok=True)
    os.system(f"sudo cp -rL {jt.flags.jittor_path} {centos_path + '/src/'}")
    os.system(f"sudo cp -rL {jt.flags.jittor_path}/../jittor_utils {centos_path + '/src/'}")
    run_cmd(f'sudo docker build --tag centos_build_env -f /tmp/centos_build_env .')
    run_cmd(f"sudo docker run --rm -v {centos_path}:/root/.cache/jittor centos_build_env scl enable devtoolset-7 'PYTHONPATH=/root/.cache/jittor/src {env} python3.8 -m jittor.test.test_core'")
    run_cmd(f"sudo docker run --rm -v {centos_path}:/root/.cache/jittor centos_build_env scl enable devtoolset-7 'PYTHONPATH=/root/.cache/jittor/src {env} python3.8 -m jittor.test.test_core'")