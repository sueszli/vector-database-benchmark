from bson import ObjectId
from app import utils

class BaseUpdateTask(object):

    def __init__(self, task_id: str):
        if False:
            i = 10
            return i + 15
        self.task_id = task_id

    def update_services(self, service_name: str, elapsed: float):
        if False:
            while True:
                i = 10
        elapsed = '{:.2f}'.format(elapsed)
        self.update_task_field('status', service_name)
        query = {'_id': ObjectId(self.task_id)}
        update = {'$push': {'service': {'name': service_name, 'elapsed': float(elapsed)}}}
        utils.conn_db('task').update_one(query, update)

    def update_task_field(self, field=None, value=None):
        if False:
            for i in range(10):
                print('nop')
        query = {'_id': ObjectId(self.task_id)}
        update = {'$set': {field: value}}
        utils.conn_db('task').update_one(query, update)