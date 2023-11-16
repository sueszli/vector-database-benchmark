import ray
(a, b, c) = (ray.put(1), ray.put(2), ray.put(3))

@ray.remote
def print_via_capture():
    if False:
        print('Hello World!')
    'This function prints the values of (a, b, c) to stdout.'
    print(ray.get([a, b, c]))
print_via_capture.remote()