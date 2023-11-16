import json
import yaml
from yaml import CSafeLoader
from yaml import SafeLoader
from yaml import SafeLoader as NewSafeLoader

def test_yaml_load():
    if False:
        print('Hello World!')
    ystr = yaml.dump({'a': 1, 'b': 2, 'c': 3})
    y = yaml.load(ystr)
    yaml.dump(y)
    try:
        y = yaml.load(ystr, Loader=yaml.CSafeLoader)
    except AttributeError:
        y = yaml.load(ystr, Loader=yaml.SafeLoader)

def test_json_load():
    if False:
        i = 10
        return i + 15
    j = json.load('{}')
yaml.load('{}', Loader=yaml.Loader)
yaml.load('{}', SafeLoader)
yaml.load('{}', yaml.SafeLoader)
yaml.load('{}', CSafeLoader)
yaml.load('{}', yaml.CSafeLoader)
yaml.load('{}', NewSafeLoader)