obj = {}
key in obj.keys()
key not in obj.keys()
foo['bar'] in obj.keys()
foo['bar'] not in obj.keys()
foo['bar'] in obj.keys()
foo['bar'] not in obj.keys()
foo() in obj.keys()
foo() not in obj.keys()
for key in obj.keys():
    pass
for key in list(obj.keys()):
    if some_property(key):
        del obj[key]
[k for k in obj.keys()]
{k for k in obj.keys()}
{k: k for k in obj.keys()}
(k for k in obj.keys())
key in (obj or {}).keys()
key in (obj or {}).keys()
from typing import KeysView

class Foo:

    def keys(self) -> KeysView[object]:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __contains__(self, key: object) -> bool:
        if False:
            while True:
                i = 10
        return key in self.keys()
key in obj.keys() and foo
key in obj.keys() and foo
key in obj.keys() and foo
for key in self.experiment.surveys[0].stations[0].keys():
    continue