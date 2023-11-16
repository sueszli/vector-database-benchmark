import argparse
import contextlib
import time
import cupy
import numpy

def black_scholes(xp, s, x, t, r, v):
    if False:
        return 10
    sqrt_t = xp.sqrt(t)
    d1 = (xp.log(s / x) + (r + v * v / 2) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t

    def get_cumulative_normal_distribution(x):
        if False:
            i = 10
            return i + 15
        A1 = 0.31938153
        A2 = -0.356563782
        A3 = 1.781477937
        A4 = -1.821255978
        A5 = 1.330274429
        RSQRT2PI = 0.3989422804014327
        W = 0.2316419
        k = 1 / (1 + W * xp.abs(x))
        cnd = RSQRT2PI * xp.exp(-x * x / 2) * (k * (A1 + k * (A2 + k * (A3 + k * (A4 + k * A5)))))
        cnd = xp.where(x > 0, 1 - cnd, cnd)
        return cnd
    cnd_d1 = get_cumulative_normal_distribution(d1)
    cnd_d2 = get_cumulative_normal_distribution(d2)
    exp_rt = xp.exp(-r * t)
    call = s * cnd_d1 - x * exp_rt * cnd_d2
    put = x * exp_rt * (1 - cnd_d2) - s * (1 - cnd_d1)
    return (call, put)
black_scholes_kernel = cupy.ElementwiseKernel('T s, T x, T t, T r, T v', 'T call, T put', '\n    const T sqrt_t = sqrt(t);\n    const T d1 = (log(s / x) + (r + v * v / 2) * t) / (v * sqrt_t);\n    const T d2 = d1 - v * sqrt_t;\n\n    const T cnd_d1 = get_cumulative_normal_distribution(d1);\n    const T cnd_d2 = get_cumulative_normal_distribution(d2);\n\n    const T exp_rt = exp(- r * t);\n    call = s * cnd_d1 - x * exp_rt * cnd_d2;\n    put = x * exp_rt * (1 - cnd_d2) - s * (1 - cnd_d1);\n    ', 'black_scholes_kernel', preamble='\n    __device__\n    inline T get_cumulative_normal_distribution(T x) {\n        const T A1 = 0.31938153;\n        const T A2 = -0.356563782;\n        const T A3 = 1.781477937;\n        const T A4 = -1.821255978;\n        const T A5 = 1.330274429;\n        const T RSQRT2PI = 0.39894228040143267793994605993438;\n        const T W = 0.2316419;\n\n        const T k = 1 / (1 + W * abs(x));\n        T cnd = RSQRT2PI * exp(- x * x / 2) *\n            (k * (A1 + k * (A2 + k * (A3 + k * (A4 + k * A5)))));\n        if (x > 0) {\n            cnd = 1 - cnd;\n        }\n        return cnd;\n    }\n    ')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', '-g', default=0, type=int, help='GPU ID')
    parser.add_argument('--n-options', '-n', default=10000000, type=int)
    args = parser.parse_args()
    cupy.cuda.Device(args.gpu_id).use()

    def rand_range(m, M):
        if False:
            for i in range(10):
                print('nop')
        samples = cupy.random.rand(args.n_options)
        return (m + (M - m) * samples).astype(numpy.float64)
    print('initializing...')
    stock_price_gpu = rand_range(5, 30)
    option_strike_gpu = rand_range(1, 100)
    option_years_gpu = rand_range(0.25, 10)
    stock_price_cpu = stock_price_gpu.get()
    option_strike_cpu = option_strike_gpu.get()
    option_years_cpu = option_years_gpu.get()

    @contextlib.contextmanager
    def timer(message):
        if False:
            for i in range(10):
                print('nop')
        cupy.cuda.Stream.null.synchronize()
        start = time.time()
        yield
        cupy.cuda.Stream.null.synchronize()
        end = time.time()
        print('%s:\t%f sec' % (message, end - start))
    print('start computation')
    risk_free = 0.02
    volatility = 0.3
    with timer(' CPU (NumPy, Naive implementation)'):
        (call_cpu, put_cpu) = black_scholes(numpy, stock_price_cpu, option_strike_cpu, option_years_cpu, risk_free, volatility)
    with timer(' GPU (CuPy, Naive implementation)'):
        (call_gpu1, put_gpu1) = black_scholes(cupy, stock_price_gpu, option_strike_gpu, option_years_gpu, risk_free, volatility)
    with timer(' GPU (CuPy, Elementwise kernel)'):
        (call_gpu2, put_gpu2) = black_scholes_kernel(stock_price_gpu, option_strike_gpu, option_years_gpu, risk_free, volatility)
    cupy.testing.assert_allclose(call_cpu, call_gpu1)
    cupy.testing.assert_allclose(call_cpu, call_gpu2)
    cupy.testing.assert_allclose(put_cpu, put_gpu1)
    cupy.testing.assert_allclose(put_cpu, put_gpu2)