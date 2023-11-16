from django.db.models.expressions import Func

class Position(Func):
    function = 'POSITION'
    template = "%(function)s('%(substring)s' in %(expressions)s)"

    def __init__(self, expression, substring):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(expression, substring=substring)