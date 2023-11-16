import argparse
import torch

def add_parser_arguments(parser):
    if False:
        for i in range(10):
            print('nop')
    parser.add_argument('--checkpoint-path', metavar='<path>', help='checkpoint filename')
    parser.add_argument('--weight-path', metavar='<path>', help='name of file in which to store weights')
    parser.add_argument('--ema', action='store_true', default=False)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    add_parser_arguments(parser)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    key = 'state_dict' if not args.ema else 'ema_state_dict'
    model_state_dict = {k[len('module.'):] if 'module.' in k else k: v for (k, v) in checkpoint['state_dict'].items()}
    print(f"Loaded model, acc : {checkpoint['best_prec1']}")
    torch.save(model_state_dict, args.weight_path)