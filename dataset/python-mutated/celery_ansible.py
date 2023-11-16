from .celery_base import CeleryBaseService
__all__ = ['CeleryAnsibleService']

class CeleryAnsibleService(CeleryBaseService):

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        kwargs['queue'] = 'ansible'
        super().__init__(**kwargs)