def cook(dat):
    if False:
        for i in range(10):
            print('nop')
    dat.clear()
    dat.copy(dat.inputs[0])
    dat.appendCol(['module'])
    dat.appendCol(['localname'])
    for pname in dat.col('name')[1:]:
        if ':' in pname.val:
            (module, localname) = pname.val.split(':')
            dat[pname, 'module'] = module
            dat[pname, 'localname'] = localname