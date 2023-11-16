try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader
try:
    from yaml import CSafeDumper as Dumper
except ImportError:
    from yaml import SafeDumper as Dumper
YamlDumper = Dumper

class YamlLoader(Loader):

    def construct_mapping(self, node, deep=False):
        if False:
            while True:
                i = 10
        mapping = []
        for (key_node, value_node) in node.value:
            key = self.construct_object(key_node, deep=deep)
            assert key not in mapping, f'Found a duplicate key in the yaml. key={key}, line={node.start_mark.line}'
            mapping.append(key)
        mapping = super().construct_mapping(node, deep=deep)
        return mapping