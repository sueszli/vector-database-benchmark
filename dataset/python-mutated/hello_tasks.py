from prefect import flow, get_run_logger, task

@task
def say_hello(name: str):
    if False:
        while True:
            i = 10
    get_run_logger().info(f'Hello {name}!')

@flow
def hello(name: str='world', count: int=1):
    if False:
        return 10
    say_hello.map((f'{name}-{i}' for i in range(count)))
if __name__ == '__main__':
    hello(count=3)