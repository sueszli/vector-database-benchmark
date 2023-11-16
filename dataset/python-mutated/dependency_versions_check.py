from .dependency_versions_table import deps
from .utils.versions import require_version, require_version_core
pkgs_to_check_at_runtime = ['python', 'tqdm', 'regex', 'requests', 'packaging', 'filelock', 'numpy', 'tokenizers', 'huggingface-hub', 'safetensors', 'accelerate', 'pyyaml']
for pkg in pkgs_to_check_at_runtime:
    if pkg in deps:
        if pkg == 'tokenizers':
            from .utils import is_tokenizers_available
            if not is_tokenizers_available():
                continue
        elif pkg == 'accelerate':
            from .utils import is_accelerate_available
            if not is_accelerate_available():
                continue
        require_version_core(deps[pkg])
    else:
        raise ValueError(f"can't find {pkg} in {deps.keys()}, check dependency_versions_table.py")

def dep_version_check(pkg, hint=None):
    if False:
        for i in range(10):
            print('nop')
    require_version(deps[pkg], hint)