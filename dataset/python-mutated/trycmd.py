def trycmd(config):
    if False:
        for i in range(10):
            print('nop')
    from buildbot.clients import tryclient
    t = tryclient.Try(config)
    t.run()
    return 0