import yaml

def _construct_yaml_str(self, node):
    if False:
        for i in range(10):
            print('nop')
    return self.construct_scalar(node)
yaml.Loader.add_constructor('tag:yaml.org,2002:str', _construct_yaml_str)
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:str', _construct_yaml_str)