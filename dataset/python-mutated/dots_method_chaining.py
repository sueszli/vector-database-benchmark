def foo():
    if False:
        return 10
    f = o.foo().m().h().bar().z()
    f = o.foo().bar()
    f = o.foo().m().h().z()
    f = o.before().foo().m().h().bar().z()