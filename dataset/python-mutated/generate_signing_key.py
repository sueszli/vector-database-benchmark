import argparse
import sys
from signedjson.key import generate_signing_key, write_signing_keys
from synapse.util.stringutils import random_string

def main() -> None:
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_file', type=argparse.FileType('w'), default=sys.stdout, help='Where to write the output to')
    args = parser.parse_args()
    key_id = 'a_' + random_string(4)
    key = (generate_signing_key(key_id),)
    write_signing_keys(args.output_file, key)
if __name__ == '__main__':
    main()