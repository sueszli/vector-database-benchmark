"""
This example demonstrates how Python code can be run in a browser, which
is faster than CPython in many cases. We run the exact same code to find
the n-th prime on both Python and JS and measure the performance.
"""
from time import perf_counter
from flexx import app, event

def find_prime(n):
    if False:
        for i in range(10):
            print('nop')
    primes = []

    def isprime(x):
        if False:
            for i in range(10):
                print('nop')
        if x <= 1:
            return False
        elif x == 2:
            return True
        for i in range(2, x // 2 + 1):
            if x % i == 0:
                return False
        return True
    t0 = perf_counter()
    i = 0
    while len(primes) < n:
        i += 1
        if isprime(i):
            primes.append(i)
    t1 = perf_counter()
    print(i, 'found in ', t1 - t0, 'seconds')

class PrimeFinder(app.PyComponent):

    def init(self):
        if False:
            print('Hello World!')
        self.js = PrimeFinderJs()

    @event.action
    def find_prime_py(self, n):
        if False:
            i = 10
            return i + 15
        find_prime(n)

    @event.action
    def find_prime_js(self, n):
        if False:
            while True:
                i = 10
        self.js.find_prime_js(n)

class PrimeFinderJs(app.JsComponent):

    @event.action
    def find_prime_js(self, n):
        if False:
            return 10
        find_prime(n)
if __name__ == '__main__':
    finder = app.launch(PrimeFinder, 'app or chrome-app')
    finder.find_prime_py(2000)
    finder.find_prime_js(2000)
    app.run()