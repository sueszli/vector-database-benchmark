import yaml

class CatalogDumper(yaml.Dumper):

    def increase_indent(self, flow=False, indentless=False):
        if False:
            while True:
                i = 10
        return super(CatalogDumper, self).increase_indent(flow, False)