from ray import serve
1 / 0

@serve.deployment(ray_actor_options={'num_cpus': 0.1})
def f(*args):
    if False:
        while True:
            i = 10
    return 'hello world'
app = f.bind()