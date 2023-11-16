import ray

@ray.remote
def echo_and_get(x_list):
    if False:
        i = 10
        return i + 15
    'This function prints its input values to stdout.'
    print('args:', x_list)
    print('values:', ray.get(x_list))
(a, b, c) = (ray.put(1), ray.put(2), ray.put(3))
echo_and_get.remote([a, b, c])