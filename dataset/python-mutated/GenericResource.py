from mage_ai.api.resources.BaseResource import BaseResource

class GenericResource(BaseResource):

    def __getattr__(self, name):
        if False:
            while True:
                i = 10

        def _missing(*args, **kwargs):
            if False:
                while True:
                    i = 10
            if type(self.model) is dict:
                return self.model.get(name)
            return self.model
        return _missing()