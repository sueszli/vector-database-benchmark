from __future__ import absolute_import
import uuid
from st2common.runners.base_action import Action
__all__ = ['GenerateUUID']

class GenerateUUID(Action):

    def run(self, uuid_type):
        if False:
            print('Hello World!')
        if uuid_type == 'uuid1':
            return str(uuid.uuid1())
        elif uuid_type == 'uuid4':
            return str(uuid.uuid4())
        else:
            raise ValueError('Unknown uuid_type. Only uuid1 and uuid4 are supported')