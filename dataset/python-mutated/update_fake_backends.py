"""Utility script to update fake backends"""
import argparse
from datetime import datetime
import json
import os
from qiskit.providers.ibmq import IBMQ
from qiskit.circuit.parameterexpression import ParameterExpression

class BackendEncoder(json.JSONEncoder):
    """A json encoder for qobj"""

    def default(self, o):
        if False:
            print('Hello World!')
        if hasattr(o, 'tolist'):
            return o.tolist()
        if isinstance(o, complex):
            return [o.real, o.imag]
        if isinstance(o, ParameterExpression):
            return float(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'qiskit', 'providers', 'fake_provider', 'backends')

def _main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Generate fake backend snapshots')
    parser.add_argument('--dir', '-d', type=str, default=DEFAULT_DIR)
    parser.add_argument('backends', type=str, nargs='*')
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--hub', type=str, default=None)
    parser.add_argument('--group', type=str, default=None)
    args = parser.parse_args()
    provider = IBMQ.load_account()
    if args.hub or args.group or args.project:
        provider = IBMQ.get_provider(hub=args.hub, group=args.group, project=args.project)
    ibmq_backends = provider.backends()
    for backend in ibmq_backends:
        raw_name = backend.name()
        if 'sim' in raw_name:
            continue
        if raw_name == 'ibmqx2':
            name = 'yorktown'
        else:
            name = raw_name.split('_')[1]
            if name == '16':
                name = 'melbourne'
        if not args.backends or (name in args.backends or raw_name in args.backends):
            if not os.path.isdir(os.path.join(args.dir, name)):
                print('Skipping, fake backend for %s does not exist yet' % name)
                continue
            config = backend.configuration()
            props = backend.properties()
            defs = backend.defaults()
            if config:
                config_path = os.path.join(args.dir, name, 'conf_%s.json' % name)
                config_dict = config.to_dict()
                with open(config_path, 'w') as fd:
                    fd.write(json.dumps(config_dict, cls=BackendEncoder))
            if props:
                props_path = os.path.join(args.dir, name, 'props_%s.json' % name)
                with open(props_path, 'w') as fd:
                    fd.write(json.dumps(props.to_dict(), cls=BackendEncoder))
            if defs:
                defs_path = os.path.join(args.dir, name, 'defs_%s.json' % name)
                with open(defs_path, 'w') as fd:
                    fd.write(json.dumps(defs.to_dict(), cls=BackendEncoder))
if __name__ == '__main__':
    _main()