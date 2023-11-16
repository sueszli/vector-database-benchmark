from argparse import ArgumentParser
from mmdet3d.apis import inference_mono_3d_detector, init_model, show_result_meshlab

def main():
    if False:
        for i in range(10):
            print('nop')
    parser = ArgumentParser()
    parser.add_argument('image', help='image file')
    parser.add_argument('ann', help='ann file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.15, help='bbox score threshold')
    parser.add_argument('--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument('--show', action='store_true', help='show online visualization results')
    parser.add_argument('--snapshot', action='store_true', help='whether to save online visualization results')
    args = parser.parse_args()
    model = init_model(args.config, args.checkpoint, device=args.device)
    (result, data) = inference_mono_3d_detector(model, args.image, args.ann)
    show_result_meshlab(data, result, args.out_dir, args.score_thr, show=args.show, snapshot=args.snapshot, task='mono-det')
if __name__ == '__main__':
    main()