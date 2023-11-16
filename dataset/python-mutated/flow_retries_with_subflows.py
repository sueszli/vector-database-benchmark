from prefect import flow
child_flow_run_count = 0
flow_run_count = 0

@flow
def child_flow():
    if False:
        i = 10
        return i + 15
    global child_flow_run_count
    child_flow_run_count += 1
    if flow_run_count < 2:
        raise ValueError()
    return 'hello'

@flow(retries=10)
def parent_flow():
    if False:
        return 10
    global flow_run_count
    flow_run_count += 1
    result = child_flow()
    if flow_run_count < 3:
        raise ValueError()
    return result
if __name__ == '__main__':
    result = parent_flow()
    assert result == 'hello', f'Got {result}'
    assert flow_run_count == 3, f'Got {flow_run_count}'
    assert child_flow_run_count == 2, f'Got {child_flow_run_count}'