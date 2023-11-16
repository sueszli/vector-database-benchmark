import logging
from viztracer import VizLoggingHandler, get_tracer

def fib(n):
    if False:
        i = 10
        return i + 15
    if n < 2:
        logging.warning('Base case, return 1')
        return 1
    logging.info(f'Recursive, working on {n}')
    return fib(n - 1) + fib(n - 2)
handler = VizLoggingHandler()
handler.setTracer(get_tracer())
logging.basicConfig(handlers=[handler], level=logging.INFO)
fib(7)