import json
import sys
import nni

def main():
    if False:
        print('Hello World!')
    data = {}
    data['nniVersion'] = nni.__version__
    data['nniPath'] = nni.__path__[0]
    data['pythonVersion'] = '{}.{}.{}-{}-{}'.format(*sys.version_info)
    data['pythonPath'] = sys.executable
    print(json.dumps(data))
if __name__ == '__main__':
    main()