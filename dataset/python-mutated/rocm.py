import spack.variant
from spack.directives import conflicts, depends_on, variant
from spack.package_base import PackageBase

class ROCmPackage(PackageBase):
    """Auxiliary class which contains ROCm variant, dependencies and conflicts
    and is meant to unify and facilitate its usage. Closely mimics CudaPackage.

    Maintainers: dtaller
    """
    amdgpu_targets = ('gfx701', 'gfx801', 'gfx802', 'gfx803', 'gfx900', 'gfx900:xnack-', 'gfx902', 'gfx904', 'gfx906', 'gfx906:xnack-', 'gfx908', 'gfx908:xnack-', 'gfx909', 'gfx90a', 'gfx90a:xnack-', 'gfx90a:xnack+', 'gfx90c', 'gfx940', 'gfx1010', 'gfx1011', 'gfx1012', 'gfx1013', 'gfx1030', 'gfx1031', 'gfx1032', 'gfx1033', 'gfx1034', 'gfx1035', 'gfx1036', 'gfx1100', 'gfx1101', 'gfx1102', 'gfx1103')
    variant('rocm', default=False, description='Enable ROCm support')
    variant('amdgpu_target', description='AMD GPU architecture', values=spack.variant.any_combination_of(*amdgpu_targets), sticky=True, when='+rocm')
    depends_on('llvm-amdgpu', when='+rocm')
    depends_on('hsa-rocr-dev', when='+rocm')
    depends_on('hip +rocm', when='+rocm')
    conflicts('amdgpu_target=none', when='+rocm')

    @staticmethod
    def hip_flags(amdgpu_target):
        if False:
            return 10
        archs = ','.join(amdgpu_target)
        return '--amdgpu-target={0}'.format(archs)
    depends_on('llvm-amdgpu@4.1.0:', when='amdgpu_target=gfx900:xnack-')
    depends_on('llvm-amdgpu@4.1.0:', when='amdgpu_target=gfx906:xnack-')
    depends_on('llvm-amdgpu@4.1.0:', when='amdgpu_target=gfx908:xnack-')
    depends_on('llvm-amdgpu@4.1.0:', when='amdgpu_target=gfx90c')
    depends_on('llvm-amdgpu@4.3.0:', when='amdgpu_target=gfx90a')
    depends_on('llvm-amdgpu@4.3.0:', when='amdgpu_target=gfx90a:xnack-')
    depends_on('llvm-amdgpu@4.3.0:', when='amdgpu_target=gfx90a:xnack+')
    depends_on('llvm-amdgpu@5.2.0:', when='amdgpu_target=gfx940')
    depends_on('llvm-amdgpu@4.5.0:', when='amdgpu_target=gfx1013')
    depends_on('llvm-amdgpu@3.8.0:', when='amdgpu_target=gfx1030')
    depends_on('llvm-amdgpu@3.9.0:', when='amdgpu_target=gfx1031')
    depends_on('llvm-amdgpu@4.1.0:', when='amdgpu_target=gfx1032')
    depends_on('llvm-amdgpu@4.1.0:', when='amdgpu_target=gfx1033')
    depends_on('llvm-amdgpu@4.3.0:', when='amdgpu_target=gfx1034')
    depends_on('llvm-amdgpu@4.5.0:', when='amdgpu_target=gfx1035')
    depends_on('llvm-amdgpu@5.2.0:', when='amdgpu_target=gfx1036')
    depends_on('llvm-amdgpu@5.3.0:', when='amdgpu_target=gfx1100')
    depends_on('llvm-amdgpu@5.3.0:', when='amdgpu_target=gfx1101')
    depends_on('llvm-amdgpu@5.3.0:', when='amdgpu_target=gfx1102')
    depends_on('llvm-amdgpu@5.3.0:', when='amdgpu_target=gfx1103')