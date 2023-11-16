def long_has_args(opt, longopts):
    if False:
        print('Hello World!')
    return [o for o in longopts if o.startswith(opt)]