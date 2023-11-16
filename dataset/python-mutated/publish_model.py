import argparse
import subprocess
import torch

def parse_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    args = parser.parse_args()
    return args

def process_checkpoint(in_file, out_file):
    if False:
        print('Hello World!')
    checkpoint = torch.load(in_file, map_location='cpu')
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    final_file = out_file.rstrip('.pth') + '-{}.pth'.format(sha[:8])
    subprocess.Popen(['mv', out_file, final_file])

def main():
    if False:
        while True:
            i = 10
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file)
if __name__ == '__main__':
    main()