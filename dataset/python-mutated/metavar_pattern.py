def foo():
    if False:
        return 10
    try:
        foo()
    except ValidationError as e:
        return e