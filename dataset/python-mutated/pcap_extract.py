"""
This script takes in CSV files generated from wireshark's "Export Packet Dissections > as CSV" option.

And prints out nice output.
"""
import argparse
import csv
FIELDS = {'status': 0, 'transaction_id': 1, 'packet_num': (2, 4), 'proto_type': 4, 'data_size': 5, 'class': 6, 'command': 7, 'params': (8, 24)}

def expand_payload(payload):
    if False:
        print('Hello World!')
    '\n    Converts a payload string to a nice format\n\n    :param payload: String of hex digits\n    :type payload: str\n\n    :return: Dictionary\n    :rtype dict\n    '
    chunks = [payload[i:i + 2] for i in range(0, len(payload), 2)]
    result = {}
    for (header, num_range) in FIELDS.items():
        if isinstance(num_range, tuple):
            result[header] = ''.join(chunks[num_range[0]:num_range[1]])
        else:
            result[header] = chunks[num_range]
    return result

def parse_args():
    if False:
        while True:
            i = 10
    '\n    Parses command line arguments\n\n    :return: Argparse arguments object\n    '
    parser = argparse.ArgumentParser(description='Extracts info from CSV')
    parser.add_argument('file', metavar='FILE', type=str, help='CSV file')
    return parser.parse_args()

def run():
    if False:
        return 10
    '\n    Main function\n    '
    args = parse_args()
    data = []
    with open(args.file, 'r', newline='') as csv_file:
        csv_object = csv.DictReader(csv_file, delimiter=',', quotechar='"')
        for line in csv_object:
            if line['Leftover Capture Data'] == '':
                continue
            data.append(expand_payload(line['Leftover Capture Data']))
    format_string = '{0:<6}  {1:<2}  {2:<10}  {3:<4}  {4:<5} {5:<7}  {6}'
    print(format_string.format('Status', 'ID', 'Packet Num', 'Size', 'Class', 'Command', 'Params'))
    for frame in data:
        print(format_string.format(frame['status'], frame['transaction_id'], frame['packet_num'], frame['data_size'], frame['class'], frame['command'], frame['params']))
    print('')
if __name__ == '__main__':
    run()