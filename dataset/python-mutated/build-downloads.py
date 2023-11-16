import glob
import json
import os
import sys
VALID_FEATURES = {'BLE', 'CAN', 'Ethernet', 'LoRa', 'USB', 'USB-C', 'WiFi', 'Dual-core', 'External Flash', 'External RAM', 'Feather', 'JST-PH', 'JST-SH', 'mikroBUS', 'microSD', 'SDCard', 'Environment Sensor', 'IMU', 'Audio Codec', 'Battery Charging', 'Camera', 'DAC', 'Display', 'Microphone', 'PoE', 'RGB LED', 'Secure Element'}

def main(repo_path, output_path):
    if False:
        for i in range(10):
            print('nop')
    boards_index = []
    board_ids = set()
    for board_json in glob.glob(os.path.join(repo_path, 'ports/*/boards/*/board.json')):
        board_dir = os.path.dirname(board_json)
        port_dir = os.path.dirname(os.path.dirname(board_dir))
        with open(board_json, 'r') as f:
            blob = json.load(f)
            features = set(blob.get('features', []))
            if not features.issubset(VALID_FEATURES):
                print(board_json, 'unknown features:', features.difference(VALID_FEATURES), file=sys.stderr)
                sys.exit(1)
            blob['id'] = blob.get('id', os.path.basename(board_dir))
            if blob['id'] in board_ids:
                print("Duplicate board ID: '{}'".format(blob['id']), file=sys.stderr)
            board_ids.add(blob['id'])
            blob['port'] = os.path.basename(port_dir)
            blob['build'] = os.path.basename(board_dir)
            boards_index.append(blob)
        board_markdown = os.path.join(board_dir, 'board.md')
        with open(os.path.join(output_path, blob['id'] + '.md'), 'w') as f:
            if os.path.exists(board_markdown):
                with open(board_markdown, 'r') as fin:
                    f.write(fin.read())
            if blob['deploy']:
                f.write('\n\n## Installation instructions\n')
            for deploy in blob['deploy']:
                with open(os.path.join(board_dir, deploy), 'r') as fin:
                    f.write(fin.read())
    with open(os.path.join(output_path, 'index.json'), 'w') as f:
        json.dump(boards_index, f, indent=4, sort_keys=True)
        f.write('\n')
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])