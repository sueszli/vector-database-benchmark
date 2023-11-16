from ray import serve

def connect_to_db(*args, **kwargs):
    if False:
        return 10
    pass

@serve.deployment(health_check_period_s=10, health_check_timeout_s=30)
class MyDeployment:

    def __init__(self, db_addr: str):
        if False:
            print('Hello World!')
        self._my_db_connection = connect_to_db(db_addr)

    def __call__(self, request):
        if False:
            i = 10
            return i + 15
        return self._do_something_cool()

    def check_health(self):
        if False:
            i = 10
            return i + 15
        if not self._my_db_connection.is_connected():
            raise RuntimeError('uh-oh, DB connection is broken.')