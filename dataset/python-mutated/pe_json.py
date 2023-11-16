import argparse
import sys
import lief
import json

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('binary', help='PE binary')
    args = parser.parse_args()
    binary = lief.parse(args.binary)
    json_data = json.loads(lief.to_json(binary))
    print(json.dumps(json_data, sort_keys=True, indent=4))
if __name__ == '__main__':
    sys.exit(main())