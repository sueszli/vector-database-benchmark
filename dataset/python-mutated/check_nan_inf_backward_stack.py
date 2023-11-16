import paddle

def main():
    if False:
        return 10
    paddle.set_flags({'FLAGS_check_nan_inf': 1, 'FLAGS_check_nan_inf_level': 0})
    cpu_place = paddle.CPUPlace()
    x = paddle.to_tensor([1, 0.0, 3], stop_gradient=False, place=cpu_place)
    y = paddle.to_tensor([0.2, 0.0, 0.5], place=cpu_place)
    z = paddle.pow(x, y)
    paddle.autograd.backward([z])
if __name__ == '__main__':
    main()