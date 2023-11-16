dict([(i, i) for i in range(3)])
dict([(i, i) for i in range(3)], z=4)

def f(x):
    if False:
        print('Hello World!')
    return x
f"{dict([(s, s) for s in 'ab'])}"
f"{dict([(s, s) for s in 'ab'])}"
f"{dict([(s, s) for s in 'ab'])}"
f"{dict([(s, f(s)) for s in 'ab'])}"
f"{dict([(s, s) for s in 'ab']) | dict([(s, s) for s in 'ab'])}"
f"{dict([(s, s) for s in 'ab']) | dict([(s, s) for s in 'ab'])}"
saved.append(dict([(k, v) for (k, v) in list(unique_instance.__dict__.items()) if k in [f.name for f in unique_instance._meta.fields]]))