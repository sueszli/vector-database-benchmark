from . import elb_name, integer_range, network_port, tags_or_list

def validate_int_to_str(x):
    if False:
        return 10
    '\n    Backward compatibility - field was int and now str.\n    Property: HealthCheck.Interval\n    Property: HealthCheck.Timeout\n    '
    if isinstance(x, int):
        return str(x)
    if isinstance(x, str):
        return str(int(x))
    raise TypeError(f'Value {x} of type {type(x)} must be either int or str')

def validate_elb_name(x):
    if False:
        i = 10
        return i + 15
    '\n    Property: LoadBalancer.LoadBalancerName\n    '
    return elb_name(x)

def validate_network_port(x):
    if False:
        print('Hello World!')
    '\n    Property: Listener.InstancePort\n    Property: Listener.LoadBalancerPort\n    '
    return network_port(x)

def validate_tags_or_list(x):
    if False:
        return 10
    '\n    Property: LoadBalancer.Tags\n    '
    return tags_or_list(x)

def validate_threshold(port):
    if False:
        while True:
            i = 10
    '\n    Property: HealthCheck.HealthyThreshold\n    Property: HealthCheck.UnhealthyThreshold\n    '
    return integer_range(2, 10)(port)