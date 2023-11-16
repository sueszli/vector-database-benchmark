""" Nameko built-in dependencies.
"""
from nameko.extensions import DependencyProvider

class Config(DependencyProvider):
    """ Dependency provider for accessing configuration values.
    """

    def get_dependency(self, worker_ctx):
        if False:
            for i in range(10):
                print('nop')
        return self.container.config.copy()