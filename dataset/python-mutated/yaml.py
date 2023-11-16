from itertools import chain
from visidata import VisiData, Progress, JsonSheet, vd

@VisiData.api
def open_yml(vd, p):
    if False:
        while True:
            i = 10
    return YamlSheet(p.name, source=p)
VisiData.open_yaml = VisiData.open_yml

class YamlSheet(JsonSheet):

    def iterload(self):
        if False:
            return 10
        yaml = vd.importExternal('yaml', 'PyYAML')

        class PrettySafeLoader(yaml.SafeLoader):

            def construct_python_tuple(self, node):
                if False:
                    i = 10
                    return i + 15
                return tuple(self.construct_sequence(node))
        PrettySafeLoader.add_constructor(u'tag:yaml.org,2002:python/tuple', PrettySafeLoader.construct_python_tuple)
        with self.source.open() as fp:
            documents = yaml.load_all(fp, PrettySafeLoader)
            self.columns = []
            self._knownKeys.clear()
            try:
                first = next(documents)
            except StopIteration:
                yield None
                return
            try:
                second = next(documents)
            except StopIteration:
                if isinstance(first, list):
                    yield from Progress(first)
                else:
                    yield first
            else:
                yield from Progress(chain([first, second], documents), total=0)