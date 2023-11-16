import argparse
import sys
from synapse.config.logger import DEFAULT_LOG_CONFIG

def main() -> None:
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-file', type=argparse.FileType('w'), default=sys.stdout, help='File to write the configuration to. Default: stdout')
    parser.add_argument('-f', '--log-file', type=str, default='/var/log/matrix-synapse/homeserver.log', help='name of the log file')
    args = parser.parse_args()
    out = args.output_file
    out.write(DEFAULT_LOG_CONFIG.substitute(log_file=args.log_file))
    out.flush()
if __name__ == '__main__':
    main()