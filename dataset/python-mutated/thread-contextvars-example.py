import contextvars
import time
import trio
request_state = contextvars.ContextVar('request_state')

def work_in_thread(msg):
    if False:
        for i in range(10):
            print('nop')
    state_value = request_state.get()
    current_user_id = state_value['current_user_id']
    time.sleep(3)
    print(f'Processed user {current_user_id} with message {msg} in a thread worker')
    state_value['msg'] = msg

async def handle_request(current_user_id):
    current_state = {'current_user_id': current_user_id, 'msg': ''}
    request_state.set(current_state)
    await trio.to_thread.run_sync(work_in_thread, f'Hello {current_user_id}')
    new_msg = current_state['msg']
    print(f'New contextvar value from worker thread for user {current_user_id}: {new_msg}')

async def main():
    async with trio.open_nursery() as nursery:
        for i in range(3):
            nursery.start_soon(handle_request, i)
trio.run(main)