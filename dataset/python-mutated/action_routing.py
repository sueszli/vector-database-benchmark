class PyPIActionPredicate:

    def __init__(self, action: str, info):
        if False:
            print('Hello World!')
        self.action_name = action

    def text(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'pypi_action = {self.action_name}'
    phash = text

    def __call__(self, context, request) -> bool:
        if False:
            return 10
        return self.action_name == request.params.get(':action', None)

def add_pypi_action_route(config, name, action, **kwargs):
    if False:
        return 10
    config.add_route(name, '/pypi', pypi_action=action, **kwargs)

def add_pypi_action_redirect(config, action, target, **kwargs):
    if False:
        i = 10
        return i + 15
    config.add_redirect('/pypi', target, pypi_action=action, **kwargs)

def includeme(config):
    if False:
        for i in range(10):
            print('nop')
    config.add_route_predicate('pypi_action', PyPIActionPredicate)
    config.add_directive('add_pypi_action_route', add_pypi_action_route, action_wrap=False)
    config.add_directive('add_pypi_action_redirect', add_pypi_action_redirect, action_wrap=False)