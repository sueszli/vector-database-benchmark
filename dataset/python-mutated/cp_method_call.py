def mk_query(user_name):
    if False:
        i = 10
        return i + 15
    query = 'SELECT user_age FROM myapp_person where user_name = {}'
    return query.format(user_name)