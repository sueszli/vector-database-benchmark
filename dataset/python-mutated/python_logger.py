from dagster import repository

def scope_logged_job():
    if False:
        for i in range(10):
            print('nop')
    import logging
    from dagster import graph, op

    @op
    def ambitious_op():
        if False:
            while True:
                i = 10
        my_logger = logging.getLogger('my_logger')
        try:
            x = 1 / 0
            return x
        except ZeroDivisionError:
            my_logger.error("Couldn't divide by zero!")
        return None

    @graph
    def thing_one():
        if False:
            for i in range(10):
                print('nop')
        ambitious_op()
    return thing_one

def scope_logged_job2():
    if False:
        i = 10
        return i + 15
    from dagster import get_dagster_logger, graph, op

    @op
    def ambitious_op():
        if False:
            return 10
        my_logger = get_dagster_logger()
        try:
            x = 1 / 0
            return x
        except ZeroDivisionError:
            my_logger.error("Couldn't divide by zero!")
        return None

    @graph
    def thing_two():
        if False:
            return 10
        ambitious_op()
    return thing_two

@repository
def python_logging_repo():
    if False:
        return 10
    return [scope_logged_job(), scope_logged_job2()]