def source_a():
    if False:
        return 10
    ...

def sink_b(arg):
    if False:
        for i in range(10):
            print('nop')
    ...

def source_c():
    if False:
        for i in range(10):
            print('nop')
    ...

def sink_d(arg):
    if False:
        return 10
    ...

def transform_x(arg):
    if False:
        i = 10
        return i + 15
    pass

def transform_y(arg):
    if False:
        while True:
            i = 10
    pass

def transform_z(arg):
    if False:
        print('Hello World!')
    pass

def taint_with_transform_x():
    if False:
        while True:
            i = 10
    alpha = source_a()
    beta = transform_x(alpha)
    return beta

def transform_x_obscure(arg):
    if False:
        print('Hello World!')
    ...

def taint_with_transform_x_obscure():
    if False:
        i = 10
        return i + 15
    alpha = source_a()
    beta = transform_x_obscure(alpha)
    return beta

def transform_x_skip_obscure(arg):
    if False:
        print('Hello World!')
    ...

def taint_with_transform_x_skip_obscure():
    if False:
        print('Hello World!')
    alpha = source_a()
    beta = transform_x_skip_obscure(alpha)
    return beta

def taint_with_transform_yx():
    if False:
        i = 10
        return i + 15
    alpha = source_a()
    beta = transform_x(alpha)
    gamma = transform_y(beta)
    return gamma

def no_source_taint_with_transform_z():
    if False:
        return 10
    alpha = source_a()
    beta = transform_z(alpha)
    return beta

def no_source_taint_with_transform_zx():
    if False:
        print('Hello World!')
    alpha = source_a()
    beta = transform_x(alpha)
    gamma = transform_z(beta)
    return gamma

def a_to_b_no_issue():
    if False:
        i = 10
        return i + 15
    alpha = source_a()
    sink_b(alpha)

def a_to_x_to_b_issue():
    if False:
        while True:
            i = 10
    alpha = source_a()
    beta = transform_x(alpha)
    sink_b(beta)

def a_to_x_to_x_b_no_issue():
    if False:
        print('Hello World!')
    alpha = source_a()
    beta = transform_x(alpha)
    gamma = transform_x(beta)
    sink_b(gamma)

def c_to_d_issue():
    if False:
        i = 10
        return i + 15
    alpha = source_c()
    sink_d(alpha)

def c_to_x_to_d_no_issue():
    if False:
        print('Hello World!')
    alpha = source_c()
    beta = transform_x(alpha)
    sink_d(beta)

def a_to_z_to_y_to_b_no_issue():
    if False:
        while True:
            i = 10
    alpha = source_a()
    beta = transform_z(alpha)
    gamma = transform_y(beta)
    sink_b(gamma)

def a_to_y_to_z_to_b_issue():
    if False:
        print('Hello World!')
    alpha = source_a()
    beta = transform_y(alpha)
    gamma = transform_z(beta)
    sink_b(gamma)

def sink_taint_with_transform_x(arg):
    if False:
        print('Hello World!')
    alpha = transform_x(arg)
    sink_b(alpha)

def no_sink_taint_with_transform_y(arg):
    if False:
        for i in range(10):
            print('nop')
    alpha = transform_y(arg)
    sink_b(alpha)

def sink_taint_with_transform_yz(arg):
    if False:
        print('Hello World!')
    alpha = transform_y(arg)
    beta = transform_z(alpha)
    sink_b(beta)

def tito_with_transform_x(arg):
    if False:
        for i in range(10):
            print('nop')
    alpha = transform_x(arg)
    return alpha

def tito_with_transform_yz(arg):
    if False:
        while True:
            i = 10
    alpha = transform_y(arg)
    beta = transform_z(alpha)
    return beta

def nested_tito_transform_x(arg):
    if False:
        print('Hello World!')
    alpha = tito_with_transform_x(arg)
    return alpha

def source_taint_via_tito_tranform_yz():
    if False:
        return 10
    alpha = source_a()
    beta = tito_with_transform_yz(alpha)
    return beta

def sink_taint_via_tito_transform_yz(arg):
    if False:
        while True:
            i = 10
    alpha = tito_with_transform_yz(arg)
    sink_b(alpha)

def ayz_to_b_issue():
    if False:
        print('Hello World!')
    alpha = source_taint_via_tito_tranform_yz()
    sink_b(alpha)

def a_to_yzb_issue():
    if False:
        return 10
    alpha = source_a()
    sink_taint_with_transform_yz(alpha)

def source_taint_with_transform_y():
    if False:
        for i in range(10):
            print('nop')
    alpha = source_a()
    beta = transform_y(alpha)
    return beta

def sink_taint_with_transform_z(arg):
    if False:
        return 10
    alpha = transform_z(arg)
    sink_b(alpha)

def ay_to_zb_issue():
    if False:
        print('Hello World!')
    alpha = source_taint_with_transform_y()
    sink_taint_with_transform_z(alpha)

def ay_to_z_to_b_issue():
    if False:
        i = 10
        return i + 15
    alpha = source_taint_with_transform_y()
    beta = transform_z(alpha)
    sink_b(beta)

def a_to_yz_to_b_issue():
    if False:
        return 10
    alpha = source_a()
    beta = tito_with_transform_yz(alpha)
    sink_b(beta)

def a_to_yz_to_d_no_issue():
    if False:
        i = 10
        return i + 15
    alpha = source_a()
    beta = tito_with_transform_yz(alpha)
    sink_d(beta)

def c_to_yz_to_b_no_issue():
    if False:
        return 10
    alpha = source_c()
    beta = tito_with_transform_yz(alpha)
    sink_b(beta)

def a_to_y_zb_issue():
    if False:
        while True:
            i = 10
    alpha = source_a()
    beta = transform_y(alpha)
    sink_taint_with_transform_z(beta)

def ax_to_zb_no_issue():
    if False:
        while True:
            i = 10
    alpha = taint_with_transform_x_obscure()
    sink_taint_with_transform_z(alpha)

def issue_with_repeated_transforms():
    if False:
        i = 10
        return i + 15
    alpha = taint_with_transform_yx()
    sink_taint_with_transform_yz(alpha)

def test_infinite_loop():
    if False:
        print('Hello World!')
    x = source_a()
    while True:
        x = tito_with_transform_x(x)
        if x:
            return x

def sanitize_source_a_tito(arg):
    if False:
        print('Hello World!')
    return arg

def sanitize_source_c_tito(arg):
    if False:
        print('Hello World!')
    return arg

def sanitize_sink_b_tito(arg):
    if False:
        print('Hello World!')
    return arg

def sanitize_sink_d_tito(arg):
    if False:
        for i in range(10):
            print('nop')
    return arg

def sanitize_a_transform_x(arg):
    if False:
        i = 10
        return i + 15
    alpha = sanitize_source_a_tito(arg)
    beta = transform_x(alpha)
    return beta

def sanitize_a_transform_x_sanitize_c_transform_y(arg):
    if False:
        while True:
            i = 10
    alpha = sanitize_a_transform_x(arg)
    beta = sanitize_source_c_tito(alpha)
    gamma = transform_y(beta)
    return gamma

def c_to_sanitize_c_to_d_no_issue():
    if False:
        return 10
    alpha = source_c()
    beta = sanitize_source_c_tito(alpha)
    sink_d(beta)

def a_to_x_to_sanitize_a_to_b_issue():
    if False:
        print('Hello World!')
    alpha = source_a()
    beta = transform_x(alpha)
    gamma = sanitize_source_a_tito(beta)
    sink_b(gamma)

def transform_x_sanitize_a_tito(arg):
    if False:
        i = 10
        return i + 15
    alpha = transform_x(arg)
    beta = sanitize_source_a_tito(alpha)
    return beta

def a_to_x_sanitize_a_to_b_issue():
    if False:
        for i in range(10):
            print('nop')
    alpha = source_a()
    beta = transform_x_sanitize_a_tito(alpha)
    sink_b(beta)

def sanitize_a_sink_b_taint(arg):
    if False:
        for i in range(10):
            print('nop')
    alpha = sanitize_source_a_tito(arg)
    sink_b(alpha)

def a_to_x_to_sanitize_a_b_issue():
    if False:
        return 10
    alpha = source_a()
    beta = transform_x(alpha)
    sanitize_a_sink_b_taint(beta)

def ax_to_sanitize_a_b_issue():
    if False:
        while True:
            i = 10
    alpha = taint_with_transform_x()
    sanitize_a_sink_b_taint(alpha)

def transform_x_sanitize_a_sink_b_taint(arg):
    if False:
        print('Hello World!')
    alpha = transform_x(arg)
    beta = sanitize_source_a_tito(alpha)
    sink_b(beta)

def a_to_x_sanitize_a_b_issue():
    if False:
        print('Hello World!')
    alpha = source_a()
    transform_x_sanitize_a_sink_b_taint(alpha)

def a_to_sanitize_a_to_x_to_b_no_issue():
    if False:
        print('Hello World!')
    alpha = source_a()
    beta = sanitize_source_a_tito(alpha)
    gamma = transform_x(beta)
    sink_b(gamma)

def sanitize_a_transform_x_tito(arg):
    if False:
        return 10
    alpha = sanitize_source_a_tito(arg)
    beta = transform_x(alpha)
    return beta

def a_to_sanitize_a_x_to_b_no_issue():
    if False:
        while True:
            i = 10
    alpha = source_a()
    beta = sanitize_a_transform_x_tito(alpha)
    sink_b(beta)

def sanitize_a_transform_x_sink_b_no_taint(arg):
    if False:
        return 10
    alpha = sanitize_source_a_tito(arg)
    beta = transform_x(alpha)
    sink_b(beta)

def sanitize_a_to_y_to_z_to_b_taint(arg):
    if False:
        print('Hello World!')
    alpha = sanitize_source_a_tito(arg)
    beta = transform_y(alpha)
    gamma = transform_z(beta)
    sink_b(gamma)

def a_to_sanitize_a_yzb_no_issue(arg):
    if False:
        i = 10
        return i + 15
    alpha = source_a()
    sanitize_a_to_y_to_z_to_b_taint(alpha)

def axy_to_sanitize_a_yzb_issue(arg):
    if False:
        return 10
    alpha = taint_with_transform_yx()
    sanitize_a_to_y_to_z_to_b_taint(alpha)

def c_to_sanitize_d_to_d_no_issue():
    if False:
        return 10
    alpha = source_c()
    beta = sanitize_sink_d_tito(alpha)
    sink_d(beta)

def a_to_x_to_sanitize_b_to_b_no_issue():
    if False:
        print('Hello World!')
    alpha = source_a()
    beta = transform_x(alpha)
    gamma = sanitize_sink_b_tito(beta)
    sink_b(gamma)

def transform_x_sanitize_b_tito(arg):
    if False:
        i = 10
        return i + 15
    alpha = transform_x(arg)
    beta = sanitize_sink_b_tito(alpha)
    return beta

def a_to_x_sanitize_b_to_b_no_issue():
    if False:
        print('Hello World!')
    alpha = source_a()
    beta = transform_x_sanitize_b_tito(alpha)
    sink_b(beta)

def source_a_transform_x_sanitize_b_taint():
    if False:
        i = 10
        return i + 15
    alpha = source_a()
    beta = transform_x(alpha)
    gamma = sanitize_sink_b_tito(beta)
    return gamma

def a_x_sanitize_b_to_b_no_issue():
    if False:
        i = 10
        return i + 15
    alpha = source_a_transform_x_sanitize_b_taint()
    sink_b(alpha)

def source_a_transform_y_transform_z_sanitize_b_no_taint():
    if False:
        for i in range(10):
            print('nop')
    alpha = source_a()
    beta = transform_y(alpha)
    gamma = transform_z(beta)
    delta = sanitize_sink_b_tito(gamma)
    return delta

def a_to_sanitize_b_to_x_to_b_issue():
    if False:
        i = 10
        return i + 15
    alpha = source_a()
    beta = sanitize_sink_b_tito(alpha)
    gamma = transform_x(beta)
    sink_b(gamma)

def sanitize_b_transform_x_tito(arg):
    if False:
        return 10
    alpha = sanitize_sink_b_tito(arg)
    beta = transform_x(alpha)
    return beta

def a_to_sanitize_b_x_to_b_issue():
    if False:
        i = 10
        return i + 15
    alpha = source_a()
    beta = sanitize_b_transform_x_tito(alpha)
    sink_b(beta)

def source_a_sanitize_b_taint():
    if False:
        return 10
    alpha = source_a()
    beta = sanitize_sink_b_tito(alpha)
    return beta

def a_sanitize_b_to_x_to_b_issue():
    if False:
        while True:
            i = 10
    alpha = source_a_sanitize_b_taint()
    beta = transform_x(alpha)
    sink_b(beta)

def a_sanitize_b_to_xb_issue():
    if False:
        for i in range(10):
            print('nop')
    alpha = source_a_sanitize_b_taint()
    sink_taint_with_transform_x(alpha)

def source_a_sanitize_b_transform_x_taint():
    if False:
        for i in range(10):
            print('nop')
    alpha = source_a()
    beta = sanitize_sink_b_tito(alpha)
    gamma = transform_x(beta)
    return gamma

def a_sanitize_b_x_to_b_issue():
    if False:
        i = 10
        return i + 15
    alpha = source_a_sanitize_b_transform_x_taint()
    sink_b(alpha)

def transform_x_tito_sanitize_b(x, y):
    if False:
        return 10
    return transform_x(x)

def no_issue_transform_x_tito_sanitize_b():
    if False:
        i = 10
        return i + 15
    alpha = source_a()
    beta = transform_x_tito_sanitize_b(alpha)
    sink_b(beta)

def transform_x_tito_sanitize_b_all_parameters(x, y):
    if False:
        return 10
    return transform_x(x)

def no_issue_transform_x_tito_sanitize_b_all_parameters():
    if False:
        while True:
            i = 10
    alpha = source_a()
    beta = transform_x_tito_sanitize_b_all_parameters(alpha)
    sink_b(beta)

def transform_x_tito_global_sanitize_b(x, y):
    if False:
        for i in range(10):
            print('nop')
    return transform_x(x)

def no_issue_transform_x_tito_global_sanitize_b():
    if False:
        i = 10
        return i + 15
    alpha = source_a()
    beta = transform_x_tito_global_sanitize_b(alpha)
    sink_b(beta)

def transform_x_tito_sanitize_return(x, y):
    if False:
        while True:
            i = 10
    return transform_x(x)

def no_issue_transform_x_tito_sanitize_return():
    if False:
        for i in range(10):
            print('nop')
    alpha = source_a()
    beta = transform_x_tito_sanitize_return(alpha)
    sink_b(beta)

def tito_propagation_source_sanitizer_1(arg):
    if False:
        i = 10
        return i + 15
    alpha = sanitize_a_transform_x_tito(arg)
    beta = sanitize_source_c_tito(alpha)
    return beta

def tito_propagation_source_sanitizer_2(arg):
    if False:
        i = 10
        return i + 15
    alpha = sanitize_source_c_tito(arg)
    beta = sanitize_a_transform_x_tito(alpha)
    return beta

def tito_propagation_source_sanitizer_3(arg):
    if False:
        i = 10
        return i + 15
    alpha = transform_x_sanitize_a_tito(arg)
    beta = sanitize_source_c_tito(alpha)
    return beta

def tito_propagation_source_sanitizer_4(arg):
    if False:
        while True:
            i = 10
    alpha = sanitize_source_c_tito(arg)
    beta = transform_x_sanitize_a_tito(alpha)
    return beta

def tito_propagation_sink_sanitizer_1(arg):
    if False:
        while True:
            i = 10
    alpha = sanitize_b_transform_x_tito(arg)
    beta = sanitize_sink_d_tito(alpha)
    return beta

def tito_propagation_sink_sanitizer_2(arg):
    if False:
        for i in range(10):
            print('nop')
    alpha = sanitize_sink_d_tito(arg)
    beta = sanitize_b_transform_x_tito(alpha)
    return beta

def tito_propagation_sink_sanitizer_3(arg):
    if False:
        return 10
    alpha = transform_x_sanitize_b_tito(arg)
    beta = sanitize_sink_d_tito(alpha)
    return beta

def tito_propagation_sink_sanitizer_4(arg):
    if False:
        return 10
    alpha = sanitize_sink_d_tito(arg)
    beta = transform_x_sanitize_b_tito(alpha)
    return beta

def transform_y_sanitize_b_tito(arg):
    if False:
        while True:
            i = 10
    alpha = transform_y(arg)
    beta = sanitize_sink_b_tito(alpha)
    return beta

def tito_propagation_source_sink_sanitizers_mixed_1(arg):
    if False:
        for i in range(10):
            print('nop')
    alpha = sanitize_a_transform_x_tito(arg)
    beta = sanitize_source_c_tito(alpha)
    gamma = sanitize_sink_d_tito(beta)
    delta = transform_y_sanitize_b_tito(gamma)
    return delta

def tito_propagation_source_sink_sanitizers_mixed_2(arg):
    if False:
        i = 10
        return i + 15
    alpha = sanitize_source_c_tito(arg)
    beta = transform_y_sanitize_b_tito(alpha)
    gamma = sanitize_a_transform_x_tito(beta)
    delta = sanitize_sink_d_tito(gamma)
    return delta

def sanitize_all(arg):
    if False:
        i = 10
        return i + 15
    ...

def transform_x_sanitize_all(arg):
    if False:
        i = 10
        return i + 15
    arg = sanitize_all(arg)
    arg = transform_x(arg)
    return arg