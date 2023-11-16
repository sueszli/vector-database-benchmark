import multiprocessing
multiprocessing.Lock()

def f():
    if False:
        return 10
    print('ok')
if __name__ == '__main__':
    ctx = multiprocessing.get_context('forkserver')
    modname = 'test.mp_preload'
    __import__(modname)
    ctx.set_forkserver_preload([modname])
    proc = ctx.Process(target=f)
    proc.start()
    proc.join()