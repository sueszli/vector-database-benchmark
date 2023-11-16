"""
ptime.py -  Precision time function made os-independent
"""
import time as systime
START_TIME = systime.time() - systime.perf_counter()

def time():
    if False:
        print('Hello World!')
    return START_TIME + systime.perf_counter()