import paddle

def static_model(x, y):
    if False:
        for i in range(10):
            print('nop')
    z = paddle.pow(x, y)
    return z

def main():
    if False:
        return 10
    paddle.enable_static()
    paddle.set_flags({'FLAGS_check_nan_inf': 1, 'FLAGS_check_nan_inf_level': 0})
    x_static = paddle.static.data(name='x_static', shape=[3], dtype='float32')
    y_static = paddle.static.data(name='y_static', shape=[3], dtype='float32')
    x_static.stop_gradient = False
    z_static = static_model(x_static, y_static)
    grads_static = paddle.static.gradients(z_static, x_static, y_static)
    exe_static = paddle.static.Executor(paddle.CPUPlace())
    exe_static.run(paddle.static.default_startup_program())
    grads_val_static = exe_static.run(paddle.static.default_main_program(), feed={'x_static': [1, 0, 3], 'y_static': [0, 0, 0]}, fetch_list=[grads_static])
if __name__ == '__main__':
    main()