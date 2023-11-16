from bigdl.chronos.benchmark import generate_forecaster, generate_data, get_CPU_info, check_nano_env
import time
import numpy as np
import argparse
import os
from scipy import stats
import psutil
import subprocess
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.common import spawn_new_process

def train(args, model_path, forecaster, train_loader, records):
    if False:
        print('Hello World!')
    '\n    train stage will record throughput.\n    '
    if args.training_processes:
        forecaster.num_processes = args.training_processes
    epochs = args.training_epochs
    forecaster.use_ipex = True if args.ipex else False
    start_time = time.time()
    forecaster.fit(train_loader, epochs=epochs)
    training_time = time.time() - start_time
    if args.framework == 'tensorflow':
        training_sample_num = epochs * sum([x.shape[0] for (x, _) in train_loader])
    else:
        training_sample_num = epochs * len(train_loader.dataset)
    forecaster.save(model_path)
    records['training_time'] = training_time
    records['training_sample_num'] = training_sample_num
    records['train_throughput'] = training_sample_num / training_time

def throughput(args, model_path, forecaster, train_loader, test_loader, records):
    if False:
        print('Hello World!')
    '\n    throughput stage will record inference throughput.\n    '
    try:
        forecaster.load(model_path)
    except:
        forecaster.fit(train_loader, epochs=1)
    if args.framework == 'tensorflow':
        inference_sample_num = sum([x.shape[0] for (x, _) in test_loader])
    else:
        inference_sample_num = len(test_loader.dataset)
    if args.quantize:
        import onnxruntime
        sess_options = onnxruntime.SessionOptions()
        if args.cores:
            sess_options.intra_op_num_threads = args.cores
            sess_options.inter_op_num_threads = args.cores
        forecaster.quantize(test_loader, framework=args.quantize_type, sess_options=sess_options, thread_num=args.cores if args.cores else None)
        print('QUANTIZATION DONE')
    if 'torch' in args.inference_framework:
        import torch
        st = time.time()
        yhat = forecaster.predict(test_loader, quantize=args.quantize)
        total_time = time.time() - st
        records['torch_infer_throughput'] = inference_sample_num / total_time
    if 'onnx' in args.inference_framework:
        if args.cores and (not args.quantize):
            forecaster.build_onnx(thread_num=args.cores)
        st = time.time()
        yhat = forecaster.predict_with_onnx(test_loader, quantize=args.quantize)
        total_time = time.time() - st
        records['onnx_infer_throughput'] = inference_sample_num / total_time
    if 'openvino' in args.inference_framework:
        if args.cores and (not args.quantize):
            forecaster.build_openvino(thread_num=args.cores)
        st = time.time()
        yhat = forecaster.predict_with_openvino(test_loader, quantize=args.quantize)
        total_time = time.time() - st
        records['openvino_infer_throughput'] = inference_sample_num / total_time
    if 'jit' in args.inference_framework:
        if args.cores:
            forecaster.build_jit(thread_num=args.cores)
        st = time.time()
        yhat = forecaster.predict_with_jit(test_loader, quantize=args.quantize)
        total_time = time.time() - st
        records['jit_infer_throughput'] = inference_sample_num / total_time

def latency(args, model_path, forecaster, train_loader, test_loader, records):
    if False:
        return 10
    '\n    latency stage will record inference latency.\n    '
    try:
        forecaster.load(model_path)
    except:
        forecaster.fit(train_loader, epochs=1)
    (latency, latency_onnx, latency_vino, latency_jit) = ([], [], [], [])
    latency_trim_portion = 0.1
    latency_percentile = [50, 90, 95, 99]
    if args.quantize:
        import onnxruntime
        sess_options = onnxruntime.SessionOptions()
        if args.cores:
            sess_options.intra_op_num_threads = args.cores
            sess_options.inter_op_num_threads = args.cores
        forecaster.quantize(test_loader, framework=args.quantize_type, sess_options=sess_options, thread_num=args.cores if args.cores else None)
        print('QUANTIZATION DONE')
    if 'torch' in args.inference_framework:
        import torch
        if args.model == 'autoformer':
            for (x, y, x_, y_) in test_loader:
                st = time.time()
                yhat = forecaster.predict((x.numpy(), y.numpy(), x_.numpy(), y_.numpy()))
                latency.append(time.time() - st)
        else:
            for (x, y) in test_loader:
                st = time.time()
                yhat = forecaster.predict(x.numpy(), quantize=args.quantize)
                latency.append(time.time() - st)
        records['torch_latency'] = stats.trim_mean(latency, latency_trim_portion)
        records['torch_percentile_latency'] = np.percentile(latency, latency_percentile)
    if 'onnx' in args.inference_framework:
        if args.cores and (not args.quantize):
            forecaster.build_onnx(thread_num=args.cores)
        for (x, y) in test_loader:
            st = time.time()
            yhat = forecaster.predict_with_onnx(x.numpy(), quantize=args.quantize)
            latency_onnx.append(time.time() - st)
        records['onnx_latency'] = stats.trim_mean(latency_onnx, latency_trim_portion)
        records['onnx_percentile_latency'] = np.percentile(latency_onnx, latency_percentile)
    if 'openvino' in args.inference_framework:
        if args.cores and (not args.quantize):
            forecaster.build_openvino(thread_num=args.cores)
        for (x, y) in test_loader:
            st = time.time()
            yhat = forecaster.predict_with_openvino(x.numpy(), quantize=args.quantize)
            latency_vino.append(time.time() - st)
        records['openvino_latency'] = stats.trim_mean(latency_vino, latency_trim_portion)
        records['openvino_percentile_latency'] = np.percentile(latency_vino, latency_percentile)
    if 'jit' in args.inference_framework:
        if args.cores:
            forecaster.build_jit(thread_num=args.cores)
        for (x, y) in test_loader:
            st = time.time()
            yhat = forecaster.predict_with_jit(x.numpy(), quantize=args.quantize)
            latency_jit.append(time.time() - st)
        records['jit_latency'] = stats.trim_mean(latency_jit, latency_trim_portion)
        records['jit_percentile_latency'] = np.percentile(latency_jit, latency_percentile)

def accuracy(args, records, forecaster, train_loader, val_loader, test_loader):
    if False:
        return 10
    '\n    evaluate stage will record model accuracy.\n    '
    if args.framework == 'torch':
        forecaster.fit(train_loader, validation_data=val_loader, epochs=args.training_epochs, validation_mode='best_epoch')
    else:
        forecaster.fit(train_loader, epochs=args.training_epochs)
    metrics = forecaster.evaluate(test_loader, multioutput='uniform_average')
    for i in range(len(metrics)):
        records[args.metrics[i]] = metrics[i]

def result(args, records):
    if False:
        print('Hello World!')
    '\n    print benchmark information\n    '
    print('>>>>>>>>>>>>> test-run information >>>>>>>>>>>>>')
    print('\x1b[1m\tModel\x1b[0m: \x1b[0;31m' + args.model + '\x1b[0m')
    print('\x1b[1m\tStage\x1b[0m: \x1b[0;31m' + args.stage + '\x1b[0m')
    print('\x1b[1m\tDataset\x1b[0m: \x1b[0;31m' + args.dataset + '\x1b[0m')
    if args.cores:
        print('\x1b[1m\tCores\x1b[0m: \x1b[0;31m' + args.core + '\x1b[0m')
    else:
        core_num = psutil.cpu_count(logical=False) * int(subprocess.getoutput('cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l'))
        print('\x1b[1m\tCores\x1b[0m: \x1b[0;31m' + core_num + '\x1b[0m')
    print('\x1b[1m\tLookback\x1b[0m: \x1b[0;31m' + args.lookback + '\x1b[0m')
    print('\x1b[1m\tHorizon\x1b[0m: \x1b[0;31m' + args.horizon + '\x1b[0m')
    if args.stage == 'train':
        print('\n>>>>>>>>>>>>> train result >>>>>>>>>>>>>')
        print(f"\x1b[1m\tavg throughput\x1b[0m: \x1b[0;34m{records['train_throughput']}\x1b[0m")
        print('>>>>>>>>>>>>> train result >>>>>>>>>>>>>')
    elif args.stage == 'latency':
        for framework in args.inference_framework:
            print('\n>>>>>>>>>>>>> {} latency result >>>>>>>>>>>>>'.format(framework))
            print('avg latency: {}ms'.format(records[framework + '_latency'] * 1000))
            print('p50 latency: {}ms'.format(records[framework + '_percentile_latency'][0] * 1000))
            print('p90 latency: {}ms'.format(records[framework + '_percentile_latency'][1] * 1000))
            print('p95 latency: {}ms'.format(records[framework + '_percentile_latency'][2] * 1000))
            print('p99 latency: {}ms'.format(records[framework + '_percentile_latency'][3] * 1000))
            print('>>>>>>>>>>>>> {} latency result >>>>>>>>>>>>>'.format(framework))
    elif args.stage == 'throughput':
        for framework in args.inference_framework:
            print('\n>>>>>>>>>>>>> {} throughput result >>>>>>>>>>>>>'.format(framework))
            print('avg throughput: {}'.format(records[framework + '_infer_throughput']))
            print('>>>>>>>>>>>>> {} throughput result >>>>>>>>>>>>>'.format(framework))
    elif args.stage == 'accuracy':
        print('\n>>>>>>>>>>>>> accuracy result >>>>>>>>>>>>>')
        for metric in args.metrics:
            print('{}: {}'.format(metric, records[metric]))
        print('>>>>>>>>>>>>> accuracy result >>>>>>>>>>>>>')

def experiment(args, records):
    if False:
        while True:
            i = 10
    path = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(path, args.ckpt)
    if args.framework == 'tensorflow':
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(args.cores)
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
    (train_loader, val_loader, test_loader) = generate_data(args)
    forecaster = generate_forecaster(args)
    if args.stage == 'train':
        train(args, model_path, forecaster, train_loader, records)
    elif args.stage == 'latency':
        latency(args, model_path, forecaster, train_loader, test_loader, records)
    elif args.stage == 'throughput':
        throughput(args, model_path, forecaster, train_loader, test_loader, records)
    elif args.stage == 'accuracy':
        accuracy(args, records, forecaster, train_loader, val_loader, test_loader)
    get_CPU_info()
    check_nano_env()
    result(args, records)

def main():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='Benchmarking Parameters')
    parser.add_argument('-m', '--model', type=str, default='tcn', metavar='', help='model name, choose from tcn/lstm/seq2seq/nbeats/autoformer, default to "tcn".')
    parser.add_argument('-s', '--stage', type=str, default='train', metavar='', help='stage name, choose from train/latency/throughput/accuracy, default to "train".')
    parser.add_argument('-d', '--dataset', type=str, default='tsinghua_electricity', metavar='', help='dataset name, choose from nyc_taxi/tsinghua_electricity/synthetic_dataset, default to "tsinghua_electricity".')
    parser.add_argument('-f', '--framework', type=str, default='torch', metavar='', help='framework name, choose from torch/tensorflow, default to "torch".')
    parser.add_argument('-c', '--cores', type=int, default=0, metavar='', help='core number, default to all physical cores.')
    parser.add_argument('-l', '--lookback', type=int, metavar='lookback', required=True, help='required, the history time steps (i.e. lookback).')
    parser.add_argument('-o', '--horizon', type=int, metavar='horizon', required=True, help='required, the output time steps (i.e. horizon).')
    parser.add_argument('--training_processes', type=int, default=1, metavar='', help='number of processes when training, default to 1.')
    parser.add_argument('--training_batchsize', type=int, default=32, metavar='', help='batch size when training, default to 32.')
    parser.add_argument('--training_epochs', type=int, default=1, metavar='', help='number of epochs when training, default to 1.')
    parser.add_argument('--inference_batchsize', type=int, default=1, metavar='', help='batch size when infering, default to 1.')
    parser.add_argument('--quantize', action='store_true', help='if use the quantized model to predict, default to False.tensorflow will support quantize later.')
    parser.add_argument('--inference_framework', nargs='+', default=['torch'], metavar='', help='predict without/with accelerator, choose from torch/onnx/openvino/jit, default to "torch" (i.e. predict without accelerator).')
    parser.add_argument('--ipex', action='store_true', help='if use ipex as accelerator for trainer, default to False.')
    parser.add_argument('--quantize_type', type=str, default='pytorch_fx', metavar='', help='quantize framework, choose from pytorch_fx/pytorch_ipex/onnxrt_qlinearops/openvino, default to "pytorch_fx".')
    parser.add_argument('--ckpt', type=str, default='checkpoints/tcn', metavar='', help='checkpoint path of a trained model, e.g. "checkpoints/tcn", default to "checkpoints/tcn".')
    parser.add_argument('--metrics', type=str, nargs='+', default=['mse', 'mae'], metavar='', help='evaluation metrics of a trained model, e.g. "mse"/"mae", default to "mse, mae".')
    parser.add_argument('--normalization', action='store_true', help='if to use normalization trick to alleviate distribution shift.')
    args = parser.parse_args()
    records = vars(args)
    models = ['tcn', 'lstm', 'seq2seq', 'nbeats', 'autoformer']
    stages = ['train', 'latency', 'throughput', 'accuracy']
    datasets = ['tsinghua_electricity', 'nyc_taxi', 'synthetic_dataset']
    frameworks = ['torch', 'tensorflow']
    quantize_types = ['pytorch_fx', 'pytorch_ipex', 'onnxrt_qlinearops', 'openvino']
    quantize_torch_types = ['pytorch_fx', 'pytorch_ipex']
    invalidInputError(args.model in models, f"-m/--model argument should be one of {models}, but get '{args.model}'")
    invalidInputError(args.stage in stages, f"-s/--stage argument should be one of {stages}, but get '{args.stage}'")
    invalidInputError(args.dataset in datasets, f"-d/--dataset argument should be one of {datasets}, but get '{{args.dataset}}'")
    invalidInputError(args.framework in frameworks, f"-f/--framework argument should be one of {frameworks}, but get '{{args.framework}}'")
    invalidInputError(args.quantize_type in quantize_types, f"--quantize_type argument should be one of {quantize_types}, but get '{{args.quantize_type}}'")
    if args.quantize and 'torch' in args.inference_framework:
        invalidInputError(args.quantize_type in quantize_torch_types, f"if inference framework is 'torch', then --quantize_type argument should be one of {{quantize_torch_types}}, but get '{{args.quantize_type}}'")
    if 'onnx' in args.inference_framework:
        args.quantize_type = 'onnxrt_qlinearops'
    elif 'openvino' in args.inference_framework:
        args.quantize_type = 'openvino'
    if args.framework == 'torch':
        import torch
        if args.cores:
            torch.set_num_threads(args.cores)
        experiment(args, records)
    elif args.cores:
        new_experiment = spawn_new_process(experiment)
        new_experiment(args, records, env_var={'OMP_NUM_THREADS': str(args.cores)})
    else:
        experiment(args, records)
if __name__ == '__main__':
    main()