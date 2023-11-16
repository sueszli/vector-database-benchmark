from collections import OrderedDict
import cimodel.data.binary_build_data as binary_build_data
import cimodel.data.simple.util.branch_filters as branch_filters
import cimodel.lib.conf_tree as conf_tree
import cimodel.lib.miniutils as miniutils

class Conf:

    def __init__(self, os, gpu_version, pydistro, parms, smoke, libtorch_variant, gcc_config_variant, libtorch_config_variant):
        if False:
            i = 10
            return i + 15
        self.os = os
        self.gpu_version = gpu_version
        self.pydistro = pydistro
        self.parms = parms
        self.smoke = smoke
        self.libtorch_variant = libtorch_variant
        self.gcc_config_variant = gcc_config_variant
        self.libtorch_config_variant = libtorch_config_variant

    def gen_build_env_parms(self):
        if False:
            i = 10
            return i + 15
        elems = [self.pydistro] + self.parms + [binary_build_data.get_processor_arch_name(self.gpu_version)]
        if self.gcc_config_variant is not None:
            elems.append(str(self.gcc_config_variant))
        if self.libtorch_config_variant is not None:
            elems.append(str(self.libtorch_config_variant))
        return elems

    def gen_docker_image(self):
        if False:
            for i in range(10):
                print('nop')
        if self.gcc_config_variant == 'gcc5.4_cxx11-abi':
            if self.gpu_version is None:
                return miniutils.quote('pytorch/libtorch-cxx11-builder:cpu')
            else:
                return miniutils.quote(f'pytorch/libtorch-cxx11-builder:{self.gpu_version}')
        if self.pydistro == 'conda':
            if self.gpu_version is None:
                return miniutils.quote('pytorch/conda-builder:cpu')
            else:
                return miniutils.quote(f'pytorch/conda-builder:{self.gpu_version}')
        docker_word_substitution = {'manywheel': 'manylinux', 'libtorch': 'manylinux'}
        docker_distro_prefix = miniutils.override(self.pydistro, docker_word_substitution)
        alt_docker_suffix = 'cuda102' if not self.gpu_version else 'rocm:' + self.gpu_version.strip('rocm') if self.gpu_version.startswith('rocm') else self.gpu_version
        docker_distro_suffix = alt_docker_suffix if self.pydistro != 'conda' else 'cuda' if alt_docker_suffix.startswith('cuda') else 'rocm'
        return miniutils.quote('pytorch/' + docker_distro_prefix + '-' + docker_distro_suffix)

    def get_name_prefix(self):
        if False:
            i = 10
            return i + 15
        return 'smoke' if self.smoke else 'binary'

    def gen_build_name(self, build_or_test, nightly):
        if False:
            return 10
        parts = [self.get_name_prefix(), self.os] + self.gen_build_env_parms()
        if nightly:
            parts.append('nightly')
        if self.libtorch_variant:
            parts.append(self.libtorch_variant)
        if not self.smoke:
            parts.append(build_or_test)
        joined = '_'.join(parts)
        return joined.replace('.', '_')

    def gen_workflow_job(self, phase, upload_phase_dependency=None, nightly=False):
        if False:
            for i in range(10):
                print('nop')
        job_def = OrderedDict()
        job_def['name'] = self.gen_build_name(phase, nightly)
        job_def['build_environment'] = miniutils.quote(' '.join(self.gen_build_env_parms()))
        if self.smoke:
            job_def['requires'] = ['update_s3_htmls']
            job_def['filters'] = branch_filters.gen_filter_dict(branches_list=['postnightly'])
        else:
            filter_branch = '/.*/'
            job_def['filters'] = branch_filters.gen_filter_dict(branches_list=[filter_branch], tags_list=[branch_filters.RC_PATTERN])
        if self.libtorch_variant:
            job_def['libtorch_variant'] = miniutils.quote(self.libtorch_variant)
        if phase == 'test':
            if not self.smoke:
                job_def['requires'] = [self.gen_build_name('build', nightly)]
            if not (self.smoke and self.os == 'macos') and self.os != 'windows':
                job_def['docker_image'] = self.gen_docker_image()
            if self.os != 'windows' and self.gpu_version:
                job_def['use_cuda_docker_runtime'] = miniutils.quote('1')
        elif self.os == 'linux' and phase != 'upload':
            job_def['docker_image'] = self.gen_docker_image()
        if phase == 'test':
            if self.gpu_version:
                if self.os == 'windows':
                    job_def['executor'] = 'windows-with-nvidia-gpu'
                else:
                    job_def['resource_class'] = 'gpu.medium'
        os_name = miniutils.override(self.os, {'macos': 'mac'})
        job_name = '_'.join([self.get_name_prefix(), os_name, phase])
        return {job_name: job_def}

    def gen_upload_job(self, phase, requires_dependency):
        if False:
            print('Hello World!')
        'Generate binary_upload job for configuration\n\n          Output looks similar to:\n\n        - binary_upload:\n            name: binary_linux_manywheel_3_7m_cu113_devtoolset7_nightly_upload\n            context: org-member\n            requires: binary_linux_manywheel_3_7m_cu113_devtoolset7_nightly_test\n            filters:\n              branches:\n                only:\n                  - nightly\n              tags:\n                only: /v[0-9]+(\\.[0-9]+)*-rc[0-9]+/\n            package_type: manywheel\n            upload_subfolder: cu113\n        '
        return {'binary_upload': OrderedDict({'name': self.gen_build_name(phase, nightly=True), 'context': 'org-member', 'requires': [self.gen_build_name(requires_dependency, nightly=True)], 'filters': branch_filters.gen_filter_dict(branches_list=['nightly'], tags_list=[branch_filters.RC_PATTERN]), 'package_type': self.pydistro, 'upload_subfolder': binary_build_data.get_processor_arch_name(self.gpu_version)})}

def get_root(smoke, name):
    if False:
        return 10
    return binary_build_data.TopLevelNode(name, binary_build_data.CONFIG_TREE_DATA, smoke)

def gen_build_env_list(smoke):
    if False:
        for i in range(10):
            print('nop')
    root = get_root(smoke, 'N/A')
    config_list = conf_tree.dfs(root)
    newlist = []
    for c in config_list:
        conf = Conf(c.find_prop('os_name'), c.find_prop('gpu'), c.find_prop('package_format'), [c.find_prop('pyver')], c.find_prop('smoke') and (not c.find_prop('os_name') == 'macos_arm64'), c.find_prop('libtorch_variant'), c.find_prop('gcc_config_variant'), c.find_prop('libtorch_config_variant'))
        newlist.append(conf)
    return newlist

def predicate_exclude_macos(config):
    if False:
        print('Hello World!')
    return config.os == 'linux' or config.os == 'windows'

def get_nightly_uploads():
    if False:
        return 10
    configs = gen_build_env_list(False)
    mylist = []
    for conf in configs:
        phase_dependency = 'test' if predicate_exclude_macos(conf) else 'build'
        mylist.append(conf.gen_upload_job('upload', phase_dependency))
    return mylist

def get_post_upload_jobs():
    if False:
        i = 10
        return i + 15
    return [{'update_s3_htmls': {'name': 'update_s3_htmls', 'context': 'org-member', 'filters': branch_filters.gen_filter_dict(branches_list=['postnightly'])}}]

def get_nightly_tests():
    if False:
        while True:
            i = 10
    configs = gen_build_env_list(False)
    filtered_configs = filter(predicate_exclude_macos, configs)
    tests = []
    for conf_options in filtered_configs:
        yaml_item = conf_options.gen_workflow_job('test', nightly=True)
        tests.append(yaml_item)
    return tests

def get_jobs(toplevel_key, smoke):
    if False:
        return 10
    jobs_list = []
    configs = gen_build_env_list(smoke)
    phase = 'build' if toplevel_key == 'binarybuilds' else 'test'
    for build_config in configs:
        if phase != 'test' or build_config.os != 'macos_arm64':
            jobs_list.append(build_config.gen_workflow_job(phase, nightly=True))
    return jobs_list

def get_binary_build_jobs():
    if False:
        print('Hello World!')
    return get_jobs('binarybuilds', False)

def get_binary_smoke_test_jobs():
    if False:
        print('Hello World!')
    return get_jobs('binarysmoketests', True)