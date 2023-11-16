def get_path_snake(some_id):
    if False:
        print('Hello World!')
    data = {'SomeId': some_id}
    return data

def get_path_shadow(id_):
    if False:
        i = 10
        return i + 15
    data = {'id': id_}
    return data

def get_query_snake(some_id):
    if False:
        i = 10
        return i + 15
    data = {'someId': some_id}
    return data

def get_query_shadow(list_):
    if False:
        i = 10
        return i + 15
    data = {'list': list_}
    return data

def get_camelcase(truthiness, order_by=None):
    if False:
        for i in range(10):
            print('nop')
    data = {'truthiness': truthiness, 'order_by': order_by}
    return data

def post_path_snake(some_id, some_other_id):
    if False:
        while True:
            i = 10
    data = {'SomeId': some_id, 'SomeOtherId': some_other_id}
    return data

def post_path_shadow(id_, round_):
    if False:
        while True:
            i = 10
    data = {'id': id_, 'reduce': round_}
    return data

def post_query_snake(some_id, some_other_id):
    if False:
        for i in range(10):
            print('nop')
    data = {'someId': some_id, 'someOtherId': some_other_id}
    return data

def post_query_shadow(id_, class_, next_):
    if False:
        print('Hello World!')
    data = {'id': id_, 'class': class_, 'next': next_}
    return data