import argparse

def main():
    if False:
        i = 10
        return i + 15
    "\n    Create code file with the following format:\n        {'audio': 'file1', 'unitA': 'file1_chnl1_units', 'unitB': 'file1_chnl2_units'}\n        {'audio': 'file2', 'unitA': 'file2_chnl1_units', 'unitB': 'file2_chnl2_units'}\n        ...\n\n    Given the input units files\n        - channel1_units_file:\n            file1|file1_chnl1_units\n            file2|file2_chnl1_units\n            ...\n        - channel2_units_file:\n            file1|file1_chnl2_units\n            file2|file2_chnl2_units\n            ...\n    "
    parser = argparse.ArgumentParser()
    parser.add_argument('channel1_units_file', type=str, help='Units of the first channel.')
    parser.add_argument('channel2_units_file', type=str, help='Units of the second channel.')
    parser.add_argument('output_file', type=str, help='Output file.')
    parser.add_argument('--channels', type=str, default='unitA,unitB', help="Comma-separated list of the channel names to create in the code(Default: 'unitA,unitB').")
    args = parser.parse_args()
    channel_names = args.channels.split(',')
    with open(args.channel1_units_file) as funit1, open(args.channel2_units_file) as funit2, open(args.output_file, 'w') as fout:
        for (line1, line2) in zip(funit1, funit2):
            (fname1, units1) = line1.strip().split('|')
            (fname2, units2) = line2.strip().split('|')
            assert len(units1.split()) == len(units2.split()), f'Mismatch units length ({len(units1.split())} vs {len(units2.split())})'
            base_fname1 = fname1[:-9]
            base_fname2 = fname2[:-9]
            assert base_fname1 == base_fname2, f'Mismatch filenames ({base_fname1} vs {base_fname2}). Expected $filename-channel1 and $filename-channel2 in two files'
            code = {'audio': base_fname1, channel_names[0]: units1, channel_names[1]: units2}
            fout.write(str(code))
            fout.write('\n')
    print(f'Codes written to {args.output_file}')
if __name__ == '__main__':
    main()