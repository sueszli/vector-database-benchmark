import timeit
import numpy as np
SETUP_CODE = '\nimport mobilenet_v1\nimport torch\n\nmodel = mobilenet_v1.mobilenet_1()\nmodel.eval()\ndata = torch.rand(1, 3, 120, 120)\n'
TEST_CODE = '\nwith torch.no_grad():\n    model(data)\n'

def main():
    if False:
        i = 10
        return i + 15
    (repeat, number) = (5, 100)
    res = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=repeat, number=number)
    res = np.array(res, dtype=np.float32)
    res /= number
    (mean, var) = (np.mean(res), np.std(res))
    print('Inference speed: {:.2f}±{:.2f} ms'.format(mean * 1000, var * 1000))
if __name__ == '__main__':
    main()