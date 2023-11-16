import argparse
import contextlib
import sys
import time
import cupy
import numpy
from black_scholes import black_scholes_kernel
monte_carlo_kernel = cupy.ElementwiseKernel('T s, T x, T t, T r, T v, int32 n_samples, int32 seed', 'T call', '\n    // We can use special variables i and _ind to get the index of the thread.\n    // In this case, we used an index as a seed of random sequence.\n    uint64_t rand_state[2];\n    init_state(rand_state, i, seed);\n\n    T call_sum = 0;\n    const T v_by_sqrt_t = v * sqrt(t);\n    const T mu_by_t = (r - v * v / 2) * t;\n\n    // compute the price of the call option with Monte Carlo method\n    for (int i = 0; i < n_samples; ++i) {\n        const T p = sample_normal(rand_state);\n        call_sum += get_call_value(s, x, p, mu_by_t, v_by_sqrt_t);\n    }\n    // convert the future value of the call option to the present value\n    const T discount_factor = exp(- r * t);\n    call = discount_factor * call_sum / n_samples;\n    ', preamble='\n    #ifndef __HIPCC__\n    typedef unsigned long long uint64_t;\n    #endif\n\n    __device__\n    inline T get_call_value(T s, T x, T p, T mu_by_t, T v_by_sqrt_t) {\n        const T call_value = s * exp(mu_by_t + v_by_sqrt_t * p) - x;\n        return (call_value > 0) ? call_value : 0;\n    }\n\n    // Initialize state\n    __device__ inline void init_state(uint64_t* a, int i, int seed) {\n        a[0] = i + 1;\n        a[1] = 0x5c721fd808f616b6 + seed;\n    }\n\n    __device__ inline uint64_t xorshift128plus(uint64_t* x) {\n        uint64_t s1 = x[0];\n        uint64_t s0 = x[1];\n        x[0] = s0;\n        s1 = s1 ^ (s1 << 23);\n        s1 = s1 ^ (s1 >> 17);\n        s1 = s1 ^ s0;\n        s1 = s1 ^ (s0 >> 26);\n        x[1] = s1;\n        return s0 + s1;\n    }\n\n    // Draw a sample from an uniform distribution in a range of [0, 1]\n    __device__ inline T sample_uniform(uint64_t* state) {\n        const uint64_t x = xorshift128plus(state);\n        // 18446744073709551615 = 2^64 - 1\n        return T(x) / T(18446744073709551615);\n    }\n\n    // Draw a sample from a normal distribution with N(0, 1)\n    __device__ inline T sample_normal(uint64_t* state) {\n        T x = sample_uniform(state);\n        T s = T(-1.4142135623730950488016887242097);  // = -sqrt(2)\n        if (x > 0.5) {\n            x = 1 - x;\n            s = -s;\n        }\n        T p = x + T(0.5);\n        return s * erfcinv(2 * p);\n    }\n    ')

def compute_option_prices(stock_price, option_strike, option_years, risk_free, volatility, n_threads_per_option, n_samples_per_thread, seed=0):
    if False:
        while True:
            i = 10
    n_options = len(stock_price)
    call_prices = cupy.empty((n_options, n_threads_per_option), dtype=numpy.float64)
    monte_carlo_kernel(stock_price[:, None], option_strike[:, None], option_years[:, None], risk_free, volatility, n_samples_per_thread, seed, call_prices)
    return call_prices.mean(axis=1)

def main():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', '-g', default=0, type=int, help='GPU ID')
    parser.add_argument('--n-options', default=1000, type=int)
    parser.add_argument('--n-samples-per-thread', default=1000, type=int)
    parser.add_argument('--n-threads-per-option', default=100000, type=int)
    args = parser.parse_args()
    cupy.cuda.Device(args.gpu_id).use()

    def rand_range(m, M):
        if False:
            while True:
                i = 10
        samples = cupy.random.rand(args.n_options)
        return (m + (M - m) * samples).astype(numpy.float64)
    print('initializing...')
    stock_price = rand_range(5, 30)
    option_strike = rand_range(1, 100)
    option_years = rand_range(0.25, 10)
    risk_free = 0.02
    volatility = 0.3

    @contextlib.contextmanager
    def timer(message):
        if False:
            print('Hello World!')
        cupy.cuda.Stream.null.synchronize()
        start = time.time()
        yield
        cupy.cuda.Stream.null.synchronize()
        end = time.time()
        print('%s:\t%f sec' % (message, end - start))
    print('start computation')
    print('    # of options: {}'.format(args.n_options))
    print('    # of samples per option: {}'.format(args.n_samples_per_thread * args.n_threads_per_option))
    with timer('GPU (CuPy, Monte Carlo method)'):
        call_mc = compute_option_prices(stock_price, option_strike, option_years, risk_free, volatility, args.n_threads_per_option, args.n_samples_per_thread)
    (call_bs, _) = black_scholes_kernel(stock_price, option_strike, option_years, risk_free, volatility)
    error = cupy.std(call_mc - call_bs)
    print('Error: %f' % error)
    return 0
if __name__ == '__main__':
    sys.exit(main())