from dagster import Config, graph, op

class AddNConfig(Config):
    n: float

@op
def add_n(config: AddNConfig, number):
    if False:
        print('Hello World!')
    return number + config.n

class MultiplyByMConfig(Config):
    m: float

@op
def multiply_by_m(config: MultiplyByMConfig, number):
    if False:
        print('Hello World!')
    return number * config.m

@graph(config={'multiply_by_m': {'config': {'m': 1.8}}, 'add_n': {'config': {'n': 32}}})
def celsius_to_fahrenheit(number):
    if False:
        while True:
            i = 10
    return multiply_by_m(add_n(number))