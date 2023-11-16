from .application import Application
default_app = None

def use_app(backend_name=None, call_reuse=True):
    if False:
        i = 10
        return i + 15
    "Get/create the default Application object\n\n    It is safe to call this function multiple times, as long as\n    backend_name is None or matches the already selected backend.\n\n    Parameters\n    ----------\n    backend_name : str | None\n        The name of the backend application to use. If not specified, Vispy\n        tries to select a backend automatically. See ``vispy.use()`` for\n        details.\n    call_reuse : bool\n        Whether to call the backend's `reuse()` function (True by default).\n        Not implemented by default, but some backends need it. For example,\n        the notebook backends need to inject some JavaScript in a notebook as\n        soon as `use_app()` is called.\n\n    "
    global default_app
    if default_app is not None:
        names = default_app.backend_name.lower().replace('(', ' ').strip(') ')
        names = [name for name in names.split(' ') if name]
        if backend_name and backend_name.lower() not in names:
            raise RuntimeError('Can only select a backend once, already using %s.' % names)
        else:
            if call_reuse:
                default_app.reuse()
            return default_app
    default_app = Application(backend_name)
    return default_app

def create():
    if False:
        return 10
    'Create the native application.'
    use_app(call_reuse=False)
    return default_app.create()

def run():
    if False:
        for i in range(10):
            print('nop')
    'Enter the native GUI event loop.'
    use_app(call_reuse=False)
    return default_app.run()

def quit():
    if False:
        return 10
    'Quit the native GUI event loop.'
    use_app(call_reuse=False)
    return default_app.quit()

def process_events():
    if False:
        i = 10
        return i + 15
    'Process all pending GUI events\n\n    If the mainloop is not running, this should be done regularly to\n    keep the visualization interactive and to keep the event system going.\n    '
    use_app(call_reuse=False)
    return default_app.process_events()