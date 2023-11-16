def callback(*args, **kwargs):
    if False:
        while True:
            i = 10
    pass
__plugin_hooks__ = {'some.ordered.callback': (callback, 10)}
__plugin_pythoncompat__ = '>=2.7,<4'