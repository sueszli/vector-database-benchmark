import os
import re
from typing import Any
import yaml
ENV_VAR_MATCHER_PATTERN = re.compile('.*\\$\\{([^}^{]+)\\}.*')

def env_var_replacer(loader: yaml.Loader, node: yaml.Node) -> Any:
    if False:
        return 10
    'Convert a YAML node to a Python object, expanding variable.\n\n    Args:\n        loader (yaml.Loader): Not used\n        node (yaml.Node): Yaml node to convert to python object\n\n    Returns:\n        Any: Python object with expanded vars.\n    '
    return os.path.expandvars(node.value)

class EnvVarLoader(yaml.SafeLoader):
    pass
EnvVarLoader.add_implicit_resolver('!environment_variable', ENV_VAR_MATCHER_PATTERN, None)
EnvVarLoader.add_constructor('!environment_variable', env_var_replacer)