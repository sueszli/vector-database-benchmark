""" Benchmarking the library on inference and training """
from transformers import HfArgumentParser, PyTorchBenchmark, PyTorchBenchmarkArguments

def main():
    if False:
        while True:
            i = 10
    parser = HfArgumentParser(PyTorchBenchmarkArguments)
    try:
        benchmark_args = parser.parse_args_into_dataclasses()[0]
    except ValueError as e:
        arg_error_msg = 'Arg --no_{0} is no longer used, please use --no-{0} instead.'
        begin_error_msg = ' '.join(str(e).split(' ')[:-1])
        full_error_msg = ''
        depreciated_args = eval(str(e).split(' ')[-1])
        wrong_args = []
        for arg in depreciated_args:
            if arg[2:] in PyTorchBenchmarkArguments.deprecated_args:
                full_error_msg += arg_error_msg.format(arg[5:])
            else:
                wrong_args.append(arg)
        if len(wrong_args) > 0:
            full_error_msg = full_error_msg + begin_error_msg + str(wrong_args)
        raise ValueError(full_error_msg)
    benchmark = PyTorchBenchmark(args=benchmark_args)
    benchmark.run()
if __name__ == '__main__':
    main()