import ray

@ray.remote
def echo(a: int, b: int, c: int):
    if False:
        while True:
            i = 10
    'This function prints its input values to stdout.'
    print(a, b, c)
echo.remote(1, 2, 3)
(a, b, c) = (ray.put(1), ray.put(2), ray.put(3))
echo.remote(a, b, c)