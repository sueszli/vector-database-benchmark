import cimodel.lib.miniutils as miniutils
from cimodel.data.simple.util.branch_filters import gen_filter_dict_exclude
from cimodel.data.simple.util.versions import MultiPartVersion
XCODE_VERSION = MultiPartVersion([12, 5, 1])

class ArchVariant:

    def __init__(self, name, custom_build_name=''):
        if False:
            while True:
                i = 10
        self.name = name
        self.custom_build_name = custom_build_name

    def render(self):
        if False:
            return 10
        extra_parts = [self.custom_build_name] if len(self.custom_build_name) > 0 else []
        return '-'.join([self.name] + extra_parts).replace('_', '-')

def get_platform(arch_variant_name):
    if False:
        print('Hello World!')
    return 'SIMULATOR' if arch_variant_name == 'x86_64' else 'OS'

class IOSJob:

    def __init__(self, xcode_version, arch_variant, is_org_member_context=True, extra_props=None):
        if False:
            while True:
                i = 10
        self.xcode_version = xcode_version
        self.arch_variant = arch_variant
        self.is_org_member_context = is_org_member_context
        self.extra_props = extra_props

    def gen_name_parts(self):
        if False:
            return 10
        version_parts = self.xcode_version.render_dots_or_parts('-')
        build_variant_suffix = self.arch_variant.render()
        return ['ios'] + version_parts + [build_variant_suffix]

    def gen_job_name(self):
        if False:
            print('Hello World!')
        return '-'.join(self.gen_name_parts())

    def gen_tree(self):
        if False:
            print('Hello World!')
        platform_name = get_platform(self.arch_variant.name)
        props_dict = {'name': self.gen_job_name(), 'build_environment': self.gen_job_name(), 'ios_arch': self.arch_variant.name, 'ios_platform': platform_name}
        if self.is_org_member_context:
            props_dict['context'] = 'org-member'
        if self.extra_props:
            props_dict.update(self.extra_props)
        props_dict['filters'] = gen_filter_dict_exclude()
        return [{'pytorch_ios_build': props_dict}]
WORKFLOW_DATA = [IOSJob(XCODE_VERSION, ArchVariant('x86_64'), is_org_member_context=False, extra_props={'lite_interpreter': miniutils.quote(str(int(True)))}), IOSJob(XCODE_VERSION, ArchVariant('x86_64', 'coreml'), is_org_member_context=False, extra_props={'use_coreml': miniutils.quote(str(int(True))), 'lite_interpreter': miniutils.quote(str(int(True)))})]

def get_workflow_jobs():
    if False:
        return 10
    return [item.gen_tree() for item in WORKFLOW_DATA]