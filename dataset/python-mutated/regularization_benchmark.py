"""Benchmark regularization layers.

To run benchmarks, see the following command for an example, please change the
flag to your custom value:

```
python3 -m benchmarks.layer_benchmark.regularization_benchmark     --benchmark_name=benchmark_dropout    --num_samples=2048     --batch_size=256     --jit_compile=True
```
"""
from absl import app
from absl import flags
from benchmarks.layer_benchmark.base_benchmark import LayerBenchmark
FLAGS = flags.FLAGS

def benchmark_dropout(num_samples, batch_size, jit_compile=True):
    if False:
        print('Hello World!')
    layer_name = 'Dropout'
    init_args = {'rate': 0.5}
    benchmark = LayerBenchmark(layer_name, init_args, input_shape=[256, 256, 4], jit_compile=jit_compile)
    benchmark.benchmark_predict(num_samples=num_samples, batch_size=batch_size)
    benchmark.benchmark_train(num_samples=num_samples, batch_size=batch_size)

def benchmark_gaussian_dropout(num_samples, batch_size, jit_compile=True):
    if False:
        i = 10
        return i + 15
    layer_name = 'GaussianDropout'
    init_args = {'rate': 0.5}
    benchmark = LayerBenchmark(layer_name, init_args, input_shape=[256, 256, 4], jit_compile=jit_compile)
    benchmark.benchmark_predict(num_samples=num_samples, batch_size=batch_size)
    benchmark.benchmark_train(num_samples=num_samples, batch_size=batch_size)

def benchmark_gaussian_noise(num_samples, batch_size, jit_compile=True):
    if False:
        for i in range(10):
            print('nop')
    layer_name = 'GaussianNoise'
    init_args = {'stddev': 0.5}
    benchmark = LayerBenchmark(layer_name, init_args, input_shape=[256, 256, 4], jit_compile=jit_compile)
    benchmark.benchmark_predict(num_samples=num_samples, batch_size=batch_size)
    benchmark.benchmark_train(num_samples=num_samples, batch_size=batch_size)

def benchmark_spatial_dropout1D(num_samples, batch_size, jit_compile=True):
    if False:
        while True:
            i = 10
    layer_name = 'SpatialDropout1D'
    init_args = {'rate': 0.5}
    benchmark = LayerBenchmark(layer_name, init_args, input_shape=[256, 3], jit_compile=jit_compile)
    benchmark.benchmark_predict(num_samples=num_samples, batch_size=batch_size)
    benchmark.benchmark_train(num_samples=num_samples, batch_size=batch_size)

def benchmark_spatial_dropout2D(num_samples, batch_size, jit_compile=True):
    if False:
        while True:
            i = 10
    layer_name = 'SpatialDropout2D'
    init_args = {'rate': 0.5}
    benchmark = LayerBenchmark(layer_name, init_args, input_shape=[256, 256, 3], jit_compile=jit_compile)
    benchmark.benchmark_predict(num_samples=num_samples, batch_size=batch_size)
    benchmark.benchmark_train(num_samples=num_samples, batch_size=batch_size)

def benchmark_spatial_dropout3D(num_samples, batch_size, jit_compile=True):
    if False:
        while True:
            i = 10
    layer_name = 'SpatialDropout3D'
    init_args = {'rate': 0.5}
    benchmark = LayerBenchmark(layer_name, init_args, input_shape=[32, 32, 32, 3], jit_compile=jit_compile)
    benchmark.benchmark_predict(num_samples=num_samples, batch_size=batch_size)
    benchmark.benchmark_train(num_samples=num_samples, batch_size=batch_size)
BENCHMARK_NAMES = {'benchmark_dropout': benchmark_dropout, 'benchmark_gaussian_dropout': benchmark_gaussian_dropout, 'benchmark_gaussian_noise': benchmark_gaussian_noise, 'benchmark_spatial_dropout1D': benchmark_spatial_dropout1D, 'benchmark_spatial_dropout2D': benchmark_spatial_dropout2D, 'benchmark_spatial_dropout3D': benchmark_spatial_dropout3D}

def main(_):
    if False:
        i = 10
        return i + 15
    benchmark_name = FLAGS.benchmark_name
    num_samples = FLAGS.num_samples
    batch_size = FLAGS.batch_size
    jit_compile = FLAGS.jit_compile
    if benchmark_name is None:
        for (name, benchmark_fn) in BENCHMARK_NAMES.items():
            benchmark_fn(num_samples, batch_size, jit_compile)
        return
    if benchmark_name not in BENCHMARK_NAMES:
        raise ValueError(f'Invalid benchmark name: {benchmark_name}, `benchmark_name` must be one of {BENCHMARK_NAMES.keys()}')
    benchmark_fn = BENCHMARK_NAMES[benchmark_name]
    benchmark_fn(num_samples, batch_size, jit_compile)
if __name__ == '__main__':
    app.run(main)