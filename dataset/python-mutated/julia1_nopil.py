import sys
try:
    import __builtin__ as builtins
except ImportError:
    import builtins
try:
    builtins.profile
except AttributeError:

    def profile(func):
        if False:
            print('Hello World!')
        return func
    builtins.profile = profile
'Julia set generator without optional PIL-based image drawing'
import time
(x1, x2, y1, y2) = (-1.8, 1.8, -1.8, 1.8)
(c_real, c_imag) = (-0.62772, -0.42193)

@profile
def calculate_z_serial_purepython(maxiter, zs, cs):
    if False:
        i = 10
        return i + 15
    'Calculate output list using Julia update rule'
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while abs(z) < 2 and n < maxiter:
            z = z * z + c
            n += 1
        output[i] = n
    return output

@profile
def calc_pure_python(desired_width, max_iterations):
    if False:
        for i in range(10):
            print('nop')
    'Create a list of complex coordinates (zs) and complex\n    parameters (cs), build Julia set, and display'
    x_step = float(x2 - x1) / float(desired_width)
    y_step = float(y1 - y2) / float(desired_width)
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))
    print('Length of x:', len(x))
    print('Total elements:', len(zs))
    start_time = time.process_time()
    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    end_time = time.process_time()
    secs = end_time - start_time
    sys.stdout.flush()
    sys.stderr.flush()
    output_str = 'calculate_z_serial_purepython  took ' + str(secs) + ' seconds'
    print(output_str, file=sys.stderr)
    sys.stderr.flush()
if __name__ == '__main__':
    calc_pure_python(desired_width=1000, max_iterations=300)
    sys.exit(-1)